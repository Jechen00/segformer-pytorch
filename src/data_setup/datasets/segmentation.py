#####################################
# Imports & Dependencies
#####################################
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import pandas
from PIL import Image
from pathlib import Path

import zipfile
import shutil
import subprocess

import warnings
from abc import ABC, abstractmethod
from typing import (
    Tuple, Union, Optional, Dict, Literal, 
    TypedDict, TypeAlias, Callable, Sequence
)

from src.data_setup.transforms.ops import ImageTransform, ToImageAndMask
from src.masks import is_rgb_tuple, rgb_to_idx_mask
from src.tensor_shapes import _validate_channel_size, _validate_ndim
from src.utils import transpose_list_dict, format_idxs, all_or_none
from src.ml_types import RGBTuple, IndexLike
from src.data_setup.types import SegSample, SegSampleList


#####################################
# Class Typings
#####################################
class ClassSpec(TypedDict):
    '''
    Dictionary containing specifications for a class in a semantic segmentation dataset.
    RGB values should be in [0, 255].

    Example:
        {
            'idx': 1,
            'rgb': (255, 255, 255)
        }
    '''
    idx: int
    rgb: RGBTuple

ClassInfo: TypeAlias = Dict[str, ClassSpec]


#####################################
# Base Segmentation Class
#####################################
class SegmentationDatasetBase(ABC, Dataset):
    '''
    Base class for semantic segmentation datasets.

    Dataset Notes: 
        - See `sec.data_setup.types` for details on 
          the sample dictionary types: `SegSample` and `SegSampleList`.

    Tensor and Transform Notes:
        - If images and masks are converted to tensors (from `transforms` or `to_tensor=True`):
            - Images are expected to be 3D (channels, height, width)
            - Index masks are expected to be 2D (height, width)
            - RGB masks are expected to be 3D (3, height, width) with a channel size of 3.

        - `v2.Normalize` only acts on tensors.
          If normalization is enabled (`norm_mean` and `norm_std` are provided),
          ensure that either `to_tensor=True` or the provided `transforms` output a tensor.

        - If the original dataset contains RGB masks and they are converted 
          to tensors (from `transforms` or `to_tensor=True`), 
          they will be converted to index masks using `class_info`.
          As such, `class_info` should contain all RGB colors that 
          may appear in the masks of the dataset.
          Any undefined RGB colors will either 
          be assigned the ignore index (if `ignore_encoding` is provided)
          or cause a `ValueError` to be raised.

    Args:
        class_info (ClassInfo):
            Mapping from class names to specification dictionaries (`ClassSpec`).
            Each specification dictionary contains:
                - idx (int): Class integer index.
                - rgb (RGBTuple): Class RGB tuple with values in [0, 255].
            Each class must have a unique index and unique RGB tuple.
            From this, the following attributes are created:
                - `idx_to_class`: Mapping from index to class name.
                - `class_to_idx`: Mapping from class name to index.
                - `rgb_to_class`: Mapping from RGB color to class name.
                - `class_to_rgb`: Mapping from class name to RGB color.
                - `idx_to_rgb`: Mapping from index to RGB color.
                - `rgb_to_idx`: Mapping from RGB color to index.
        geo_transforms (optional, Callable):
            Geometric transform pipeline applied to **both** the image and mask of each sample.
            This must accept and return a sample dictionary (`SegSample`).
        img_phot_transforms (optional, Callable):
            Photometric transform pipeline applied to **only** the image of each sample.
            This must accept and return a sample dictionary (`SegSample`).
        to_tensor (bool): 
            Whether to apply tensor conversions to all samples **after** 
            any provided `geo_transforms` and `img_phot_transforms`.
            Images are scaled to [0, 1] with dtype `torch.float32`.
            Masks are converted to index masks with dtype `torch.long`.
            Default is `True`.
        norm_mean (optional, Sequence[float]): 
            Sequence of means (one for each input channel) 
            used to normalize images **after** tensor conversion.
            Masks are never normalized.
            If provided, `norm_std` must also be provided.
        norm_std (optional, Sequence[float]): 
            Sequence of standard deviations (one for each input channel) 
            used to normalize images **after** tensor conversion.
            Masks are never normalized.
            If provided, `norm_mean` must also be provided.
        ignore_encoding (optional, ClassSpec):
            Encoding that defines the integer index and RGB color
            used for ignored pixels in the segmentation masks.
            The combination of index and RGB color must not conflict
            with those in `class_info`.
            If provided, the ignore index and color can be a accessed
            through the attributes `ignore_idx` and `ignore_rgb`;
            otherwise, these attributes are set to `None`.
    '''
    def __init__(
        self,
        class_info: ClassInfo,
        geo_transforms: Optional[Callable] = None,
        img_phot_transforms: Optional[Callable] = None,
        to_tensor: bool = True,
        norm_mean: Optional[Sequence[float]] = None,
        norm_std: Optional[Sequence[float]] = None,
        ignore_encoding: Optional[ClassSpec] = None
    ):
        if not all_or_none(norm_mean, norm_std):
            raise ValueError(
                'norm_mean and norm_std must either both be provided or both be None (not provided).'
            )
        
        self.class_info = class_info
        self.ignore_encoding = ignore_encoding
        self.geo_transforms = geo_transforms
        self.img_phot_transforms = img_phot_transforms
        self.to_tensor = to_tensor
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        
        # Class name, index, and color mappings
        self._make_mappings()

        # Create transform pipeline
        self._make_transform_pipeline()
        
    def __getitem__(self, idxs: IndexLike) -> Union[SegSample, SegSampleList]:
        '''
        Gets a single-sample or multi-sample dictionary containing image and mask data.
        If a transform pipeline is available, the images and masks are transformed accordingly 
        (see `_make_transform_pipeline` for details).

        Args:
            idxs (IndexLike): 
                Index or collection of indices for the samples to retrieve.
                This supports:
                    - A single integer
                    - A list of integers
                    - A ndarray of integers
                        - Must be single-element or 1D with shape (batch_size,)
                    - A tensor of integers
                        - Must be single-element or 1D with shape (batch_size,)

        Returns:
            Union[SegSample, SegSampleList]
                A single-sample or multi-sample dictionary depending on
                whether the input was a single index or a collection of indices.

                Single-Sample (SegSample) contains the following (non-exhaustive):
                    - image (ImageInput): Transformed image sample (original if no transform pipeline).
                    - mask (ImageInput): Segmentation mask for the image.

                Multi-Sample (SegSampleList) contains the following (non-exhaustive):
                    - image (List[ImageInput]): List of transformed image samples (original if no transform pipeline).
                    - mask (List[ImageInput]): List of segmentation masks for the images.
        '''
        if isinstance(idxs, int):
            # Indexing with a single integer
            return self.get_single_item(idxs)
        else:
            # Indexing with multiple integers
            idxs = format_idxs(idxs)
            items = [self.get_single_item(idx) for idx in idxs]
            return transpose_list_dict(items, mode = 'to_cols')
        
    def get_single_item(self, idx: int) -> SegSample:    
        '''
        Gets a sample dictionary containing the image and mask data for a **single** index.
        If a transform pipeline is available, the image and mask is transformed accordingly 
        (see `_make_transform_pipeline` for details).

        Args:
            idx (int): 
                Index of the sample to retrieve.
            
        Returns:
            SegSample: 
                Single-sample dictionary containing (non-exhaustive):
                    - image (ImageInput): Transformed image sample (original if no transform pipeline).
                    - mask (ImageInput): Segmentation mask for the image.
        '''
        if not isinstance(idx, int):
            raise TypeError(f'Expected integer index. Got: {type(idx)}')
         
        item = self.get_base_item(idx)
        
        if self.transform_pipeline is not None:
            # Image and mask in item are always tensors if self.to_tensor = True
            item = self.transform_pipeline(item)
            
        img, mask = item['image'], item['mask']

        if isinstance(img, torch.Tensor):
            # Validate image tensor of shape (channels, height, width)
            _validate_ndim(img, ndim = 3, context_name = 'images')

        if isinstance(mask, torch.Tensor):
            # Convert mask to index mask (training-ready)
            item['mask'] = self._format_mask(mask)

        return item
            
    def _make_mappings(self) -> None:
        '''
        Creates the following class-related mappings:
            - `idx_to_class`: Mapping from index to class name.
            - `class_to_idx`: Mapping from class name to index.
            - `rgb_to_class`: Mapping from RGB color to class name.
            - `class_to_rgb`: Mapping from class name to RGB color.
            - `idx_to_rgb`: Mapping from index to RGB color.
            - `rgb_to_idx`: Mapping from RGB color to index.
        If `ignore_encoding` was provided at initialization, 
        the ignore encoding is also included in `idx_to_rgb` and `rgb_to_idx`.
        '''
        self.class_names = [None] * len(self.class_info)

        # Create mappings based on index
        self.idx_to_class, self.class_to_idx = {}, {}
        self.rgb_to_class, self.class_to_rgb = {}, {}
        self.idx_to_rgb, self.rgb_to_idx = {}, {}
        
        for name, spec in self.class_info.items():
            idx, rgb = self._extract_class_spec(spec)
            
            # Check for duplicates
            if rgb in self.rgb_to_idx:
                raise ValueError(f"Duplicate color mapping detected for RGB color ('rgb'): {rgb}")
            if idx in self.idx_to_rgb:
                raise ValueError(f"Duplicate index mapping detected for index ('idx'): {idx}")
                
            self.idx_to_class[idx] = name
            self.class_names[idx] = name # List version of idx_to_class
            self.class_to_idx[name] = idx
            self.rgb_to_class[rgb] = name
            self.class_to_rgb[name] = rgb
            self.idx_to_rgb[idx] = rgb
            self.rgb_to_idx[rgb] = idx

        # Optionally add ignore index to rgb_to_idx, idx_to_rgb
        self._add_ignore_mapping()
            
    def _add_ignore_mapping(self) -> None:
        '''
        Sets up handling of ignore/unmapped RGB pixels for RGB-to-index mask conversions.

        If `ignore_encoding` was provided:
            - The ignore index and RGB color are added to the mappings `rgb_to_idx` and `idx_to_rgb`.
            - The ignore index is assigned to unmapped RGB pixels during conversions.

        If `ignore_encoding` was not provided:
            - A `ValueError` is raised for unmapped RGB pixels during conversions.
        '''
        if self.ignore_encoding is None:
            self.ignore_idx, self.ignore_rgb = None, None
            self._unmapped_idx = -10000 # Unmapped RGB pixels will be flagged and errored
            return

        idx, rgb = self._extract_class_spec(self.ignore_encoding)

        # Check for duplicates
        if rgb in self.rgb_to_idx:
            if idx != self.rgb_to_idx[rgb]:
                raise ValueError(f"Ignore color conflicts with the class RGB color ('rgb'): {rgb}")
        if idx in self.idx_to_rgb:
            if rgb != self.idx_to_rgb[idx]:
                raise ValueError(f"Ignore index conflicts with the class index ('idx'): {idx}")

        self.ignore_idx, self.ignore_rgb = idx, rgb
        self.rgb_to_idx[rgb] = idx
        self.idx_to_rgb[idx] = rgb

        self._unmapped_idx = idx # Unmapped RGB pixels will be ignored (assigned ignore index)

    def _make_transform_pipeline(self) -> None:    
        '''
        Creates a single torchvision transform pipeline that may include:
            1. User-provided geometric transforms (if `geo_transforms` is provided)
            2. User-provided photometric transforms (if `phot_transforms` is provided)
            3. Image tensor conversion (if `to_tensor=True`)
            4. Float32 conversion and scaling (if `to_tensor=True`)
            5. Normalization (if `norm_mean` and `norm_std` are provided)

        The final transform pipeline is stored in `self.transform_pipeline`.
        There will be no transform pipeline (`transform_pipeline` is `None`), 
        if all these conditions are met:
            - `geo_transforms` is not provided
            - `phot_transforms` is not provided
            - `to_tensor=False`
            - `norm_mean` and `norm_std` are not provided
        '''     
        pipeline = []
        if self.geo_transforms is not None:
            pipeline.append(self.geo_transforms) # Shared geometric transforms
            
        img_phot_transforms = self.img_phot_transforms
        if img_phot_transforms is not None:
            # Image-only photometric transforms
            pipeline.append(
                img_phot_transforms if isinstance(img_phot_transforms, ImageTransform)
                else ImageTransform(img_phot_transforms)
            )

        if self.to_tensor:
            # Note: ending with v2.ToDtype will result in regular tensors (not tv_tensors)
            # Note: v2.ToDtype will not change datatype or scale tv_tensors.Mask
            pipeline.extend([
                ToImageAndMask(), # Converts to tv_tensors (Image and Mask)
                v2.ToDtype(torch.float32, scale = True) # Covert image to float32 and scales
            ])

        if self.norm_mean is not None:
            pipeline.append(
                v2.Normalize(mean = self.norm_mean, std = self.norm_std)
            )
            
        self.transform_pipeline = v2.Compose(pipeline) if (len(pipeline) > 0) else None

    def _format_mask(self, mask: torch.Tensor) -> torch.Tensor:
        '''
        Ensures that a segmentation mask is an index mask.
        Specifically:
            - If `mask` is 2D (height, width), it is assumed to be an index mask 
              and returned with dtype `torch.long`.
            - If `mask` is 3D (3, height, width), it is assumed to be a RGB mask (channel size of 3)
              and is converted to an index mask with dtype `torch.long`.

        During RGB to index mask conversion, if there are unmapped pixels and no ignore encoding,
        a `ValueError` will be raised.

        Args:
            mask (torch.Tensor):
                The segmentation mask. 
                This must be a 2D (height, width) or 3D (3, height, width) tensor.

        Returns:
            torch.Tensor:
                Index mask with shape (height, width) and dtype `torch.long`.
        '''
        mask_ndim = mask.ndim
        if mask_ndim == 2:
            # Mask is already an index mask
            return mask.to(torch.long) # Shape: (height, width)
        
        elif mask_ndim == 3:
            # Mask is a RGB mask --> need to convert to index mask
            _validate_channel_size(mask, channel_size = 3, context_name = 'RGB masks')
            idx_mask = rgb_to_idx_mask(
                rgb_masks = mask,
                rgb_to_idx = self.rgb_to_idx,
                fill_idx = self._unmapped_idx
            ) # This has dtype torch.long

            # Check for unmapped pixels
            if (self.ignore_idx is None) and ((idx_mask == self._unmapped_idx).any()):
                raise ValueError(
                    'Found unmapped pixels when constructing index mask. '
                    'Ensure all RGB pixels are covered by the color-to-index mapping '
                    'or provide an ignore_encoding.'
                )
            return idx_mask # Output shape: (height, width)
        
        else:
            raise ValueError(
                'Expected mask to be a 2D or 3D tensor after transforms. '
                f'Got {mask_ndim} dimensions.'
            )

    def _extract_class_spec(self, spec: ClassSpec) -> Tuple[int, RGBTuple]:
        '''
        Extracts the index and RGB tuple from a class specification.
        The extracted values are validated as so:
            1. Checks if the index is an integer 
            2. Checks if the RGB tuple is a tuple of 3 integers in [0, 255].

        Args:
            spec (ClassSpec): 
                The class specification to extract and validate.

        Returns:
            idx (int): 
                Index from `spec`.
            rgb (RGBTuple):
                RGB tuple from `spec`.
        '''
        idx, rgb = spec['idx'], spec['rgb']

        if (type(idx) is not int):
            raise TypeError(
                f'Expected Index to be an integer. Got type {type(idx)} for {idx}'
            )
        
        if not is_rgb_tuple(rgb):
            raise TypeError(
                f'Expected RGB to be a tuple of three integers in [0, 255]. Got: {rgb}.'
            )
        
        return idx, rgb

    @abstractmethod
    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        pass

    @abstractmethod
    def get_base_item(self, idx: int) -> SegSample:
        '''
        Gets a base dictionary containing information for a single sample
        prior to applying the dataset transform pipeline.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            SegSample:
                Single-sample dictionary containing (non-exhaustive):
                    - image (ImageInput): Base image sample.
                    - mask (ImageInput): Base segmentation mask for the image.
        '''
        pass


#####################################
# Dataset Classes
#####################################
class SuperviselyPersonDataset(SegmentationDatasetBase):
    '''
    Implements a dataset class for the Supervisely Person dataset from
    https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset

    Dataset Notes:
        - This is a binary segmentation dataset with classes:
            - background: index 0, RGB color (0, 0, 0)
            - human: index 1, RGB color (255, 255, 255)

        - The Supervisely Person dataset does not have a training/validation split,
          so it is created using the Pandas and a split seed.

    Tensor and Transform Notes:
        - `v2.Normalize` only acts on tensors.
          If normalization is enabled (`norm_mean` and `norm_std` are provided),
          ensure that either `to_tensor = True` or the provided `transforms` output a tensor.

    Args:
        root (Union[str, Path]): 
            Root directory to download the dataset in.
            The dataset will be stored in `root/supervisely_person`.
        split (Literal['train', 'val']):
            Whether to construct the training dataset (`train`) or the validation dataset (`val`).
        geo_transforms (optional, Callable):
            Geometric transform pipeline applied to **both** the image and mask of each sample.
            This must accept and return a sample dictionary (`SegSample`).
        img_phot_transforms (optional, Callable):
            Photometric transform pipeline applied to **only** the image of each sample.
            This must accept and return a sample dictionary (`SegSample`).
        to_tensor (bool): 
            Whether to apply tensor conversions to all samples **after** 
            any provided `geo_transforms` and `img_phot_transforms`.
            Images are scaled to [0, 1] with dtype `torch.float32`.
            Masks are converted to index masks with dtype `torch.long`.
            Default is `True`.
        norm_mean (optional, Sequence[float]): 
            Sequence of means (one for each input channel)
            used to normalize images **after** tensor conversion.
            If provided, `norm_std` must also be provided.
        norm_std (optional, Sequence[float]): 
            Sequence of standard deviations (one for each input channel)
            used to normalize images **after** tensor conversion.
            If provided, `norm_mean` must also be provided.
        ignore_encoding (optional, ClassSpec):
            Encoding that defines the integer index and RGB color
            used for ignored pixels in the segmentation masks.
            The pair (index, RGB) must not conflict
            with `(0, (0, 0, 0))` or `(1, (255, 255, 255))`
            If provided, the ignore index and color can be a accessed
            through the attributes `ignore_idx` and `ignore_rgb`;
            otherwise, these attributes are set to `None`.
        train_frac (float): 
            The fraction of the dataset to use as the training dataset.
            The remaining fraction (`1 - train_frac`) is used as the validation dataset.
            Default is `0.9` (90% of data for training, 10% of data for validation).
        split_seed (int): 
            The random seed used to split the dataset into training/validation components.
            Changing this will change what data is in the training/validation datasets.
            Default is `0`.
    '''
    def __init__(
        self, 
        root: Union[str, Path], 
        split: Literal['train', 'val'],
        geo_transforms: Optional[Callable] = None,
        img_phot_transforms: Optional[Callable] = None,
        to_tensor: bool = True,
        norm_mean: Optional[Sequence[float]] = None,
        norm_std: Optional[Sequence[float]] = None,
        ignore_encoding: Optional[ClassSpec] = None,
        train_frac: float = 0.9,
        split_seed: int = 0
    ):
        self.root = Path(root)
        self.data_dir = self.root / 'supervisely_person'
        self.split = split
        self.train_frac = train_frac
        self.split_seed = split_seed

        self._setup_dataset()
        class_info = {
            'human': {'idx': 1, 'rgb': (255, 255, 255)},
            'background': {'idx': 0, 'rgb': (0, 0, 0)}
        }
        
        # Initialize SegmentationDatasetBase  
        super().__init__(
            class_info = class_info,
            geo_transforms = geo_transforms,
            img_phot_transforms = img_phot_transforms,
            to_tensor = to_tensor,
            norm_mean = norm_mean,
            norm_std = norm_std,
            ignore_encoding = ignore_encoding
        )
    
    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset.
        '''
        return len(self.img_paths)
    
    def get_base_item(self, idx: int) -> SegSample:
        '''
        Gets a single-sample dictionary from the Supervisely Person dataset.
        All images and masks are converted to RGB format.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            SegSample:
                Single-sample dictionary containing (non-exhaustive):
                    - image (ImageInput): Image sample from the Supervisely Person dataset.
                                          This is a RGB PIL image.
                    - mask (ImageInput): Segmentation mask for the image.
                                         This is a RGB PIL image.
        '''
        return {
            'image': Image.open(self.img_paths[idx]).convert('RGB'),
            'mask': Image.open(self.mask_paths[idx]).convert('RGB')
        } 
    
    def _setup_dataset(self) -> None:
        '''
        Downloads and prepares the Supervisely Person dataset.

        Note: The images and masks are downloaded as PNG files.

        This involves:
            1. Downloading the dataset from 
               https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset
            2. Creating the `img_paths` and `mask_paths` attributes,
               which store the file paths to all images and masks, respectively.
        '''
        data_dir = self.data_dir
        anno_path = data_dir / 'annotations.csv' # Required annotation path
        self.anno_path = anno_path
        
        if data_dir.is_dir():
            # Dataset directory already exists
            warnings.warn(
                f'A supervisely_person directory already exists at the root {self.root}. '
                'Supervisely Person dataset will not be downloaded.',
                UserWarning
            )
            
            # Check for existence of required files and directories
            if not anno_path.is_file():
                raise FileNotFoundError(f'Dataset is missing annotation file at {anno_path}.')
                
            for dir_name in ['images', 'masks']:
                dir_path = data_dir / dir_name
                if not dir_path.is_dir():
                    raise FileNotFoundError(f'Dataset is missing {dir_name} directory at {dir_path}.')
            
        else:
            print(f'Downloading dataset zip file to {data_dir}')
            # Dataset directory doesn't exist --> download dataset zip file
            subprocess.run([
                'kaggle', 'datasets', 'download',
                '-d', 'tapakah68/supervisely-filtered-segmentation-person-dataset',
                '-p', str(data_dir)
            ], check = True)

            # Unzip file
            print(f'Unzipping dataset...')
            zip_path = data_dir / 'supervisely-filtered-segmentation-person-dataset.zip'
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            zip_path.unlink() # Remove zip file

            # Reorganize files
            (data_dir / 'df.csv').rename(anno_path)

            base_img_dir = data_dir / 'supervisely_person_clean_2667_img/supervisely_person_clean_2667_img'
            for path in base_img_dir.iterdir():
                shutil.move(path, data_dir / path.name)
            shutil.rmtree(base_img_dir.parent) # Remove original base folder

            print(f'Dataset download and extraction complete.')

        # Create img_paths and mask_paths
        self._make_img_mask_paths()
        
    def _make_img_mask_paths(self) -> None:
        '''
        Creates the `img_paths` and `mask_paths` attributes
        to store the file paths to all images and masks, respectively.
        '''
        data_dir = self.data_dir
        
        # Contains all paths to image and mask .png files
        full_anno_df = pandas.read_csv(data_dir / 'annotations.csv')
        
        # Train dataframe
        split_anno_df = full_anno_df.sample(frac = self.train_frac, random_state = self.split_seed)
        
        if self.split == 'val':
            # Validation dataframe
            split_anno_df = full_anno_df.drop(split_anno_df.index)
        
        self.img_paths = [data_dir / img_path for img_path in split_anno_df['images']]
        self.mask_paths = [data_dir / mask_path for mask_path in split_anno_df['masks']]