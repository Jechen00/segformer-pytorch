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
from typing import Tuple, Union, Optional, Dict, Literal, TypedDict, TypeAlias, Callable

from src.data_setup.transforms.ops import ImageTransform, ToImageAndMask
from src.masks import is_rgb_tuple, rgb_to_idx_mask
from src.tensor_shapes import _validate_channel_size, _validate_ndim
from src.utils import transpose_list_dict, format_idxs
from src.ml_types import RGBTuple, IndexLike
from src.data_setup.types import SegSample, SegSampleList


#####################################
# Class Typings
#####################################
class ClassSpec(TypedDict):
    idx: int
    rgb: RGBTuple

ClassInfo: TypeAlias = Dict[str, ClassSpec]


#####################################
# Base Segmentation Class
#####################################
class SegmentationDatasetBase(ABC, Dataset):
    '''
    Base class for segmentation datasets.
    '''
    def __init__(
        self,
        class_info: ClassInfo,
        geo_transforms: Optional[Callable] = None,
        img_phot_transforms: Optional[Callable] = None,
        to_tensor: bool = True,
        ignore_encoding: Optional[ClassSpec] = None
    ):
        self.class_info = class_info
        self.ignore_encoding = ignore_encoding
        self.geo_transforms = geo_transforms
        self.img_phot_transforms = img_phot_transforms
        self.to_tensor = to_tensor
        
        # Class name, index, and color mappings
        self._make_mappings()

        # Create transform pipeline
        self._make_transform_pipeline()
        
    def __getitem__(self, idxs: IndexLike) -> Union[SegSample, SegSampleList]:
        '''
        Gets a single-sample or multi-sample dictionary containing image and segmentation mask.
        Images are transformed if provided and labels are converted to tensors.

        Args:
            idxs (IndexLike): Index or collection of indices for the samples to retrieve.
                              This must be one of:
                                - A single integer
                                - A list of integers
                                - A ndarray of integers
                                - A tensor of integers

        Returns:
            A single-sample or multi-sample dictionary depending 
            whether the input was a single index or a collection of indices.

            Single-Sample (SegSample) has the keys (non-exhaustive):
                - image (ImageInput): Transformed image sample (original if no transforms).
                - mask (ImageInput): Transformed segmentation mask (original if no transforms).

            Multi-Sample (SegSampleList) has the keys (non-exhaustive):
                - image (List[ImageInput]): List of transformed image samples (original if no transforms).
                - mask (List[ImageInput]): List of transformed segmentation mask (original if no transforms).
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
        if not isinstance(idx, int):
            raise TypeError(f'Expected integer index. Got: {type(idx)}')
         
        item = self.get_raw_item(idx)
        
        if self.transform_pipeline is not None:
            # Image and mask in item are always tensors if self.to_tensor = True
            item = self.transform_pipeline(item)
            
        if self.to_tensor:
            img, mask = item['image'], item['mask']

            # Validate image tensor of shape (channels, height, width)
            _validate_ndim(img, ndim = 3, context_name = 'images')

            # Format mask to index mask (training-ready)
            item['mask'] = self._format_mask(mask)

        return item
            
    def _make_mappings(self) -> None:
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
        if self.ignore_encoding is None:
            self.ignore_idx, self.ignore_rgb = None, None
            self._unmapped_idx = -1000 # Unmapped RGB pixels will be flagged ad errors
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
            
        self.transform_pipeline = v2.Compose(pipeline) if (len(pipeline) > 0) else None

    def _format_mask(self, mask: torch.Tensor) -> torch.Tensor:
        '''
        Formats a segmentation mask to an index mask.

        If `mask` is already an index mask (2D tensor), return it unchanged.
        If `mask` is a RGB mask (3D tensor), convert it to an index mask.
        If `mask` is not a 2D or 3D tensor, raise an error.
        '''
        mask_ndim = mask.ndim
        if mask_ndim == 2:
            # Mask is already an index mask
            return mask # Shape: (height, width)
        
        elif mask_ndim == 3:
            # Mask is a RGB mask --> need to convert to index mask
            _validate_channel_size(mask, channel_size = 3, context_name = 'RGB masks')
            idx_mask = rgb_to_idx_mask(
                rgb_masks = mask,
                rgb_to_idx = self.rgb_to_idx,
                fill_idx = self._unmapped_idx
            )

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
        The extract values are validated as so:
            1. Checks if the index is an integer 
            2. Checks if the RGB tuple is a tuple of 3 integers in [0. 255].

        Args:
            spec (ClassSpec): The class specification to extract and validate

        Returns:
            idx (int): Index from `spec`.
            rgb (RGBTuple): RGB tuple from `spec`.
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
        Returns the number of samples in the dataset
        '''
        pass

    @abstractmethod
    def get_raw_item(self, idx: int) -> SegSample:
        '''
        Returns a dictionary of raw, unprocessed sample information.
        Must include 'image' and 'mask'.
        '''
        pass


#####################################
# Dataset Classes
#####################################
class SuperviselyPersonDataset(SegmentationDatasetBase):
    def __init__(
        self, 
        root: Union[str, Path], 
        split: Literal['train', 'val'],
        geo_transforms: Optional[Callable] = None,
        img_phot_transforms: Optional[Callable] = None,
        to_tensor: bool = True,
        ignore_encoding: Optional[ClassSpec] = None,
        train_frac: float = 0.8,
        split_seed: int = 0
    ):
        self.root = Path(root)
        self.data_dir = self.root / 'supervisely_person'
        self.split = split
        self.train_frac = train_frac
        self.split_seed = split_seed

        self._setup_dataset()
        
        # Initialize SegmentationDatasetBase  
        super().__init__(
            class_info = self._make_class_info(),
            geo_transforms = geo_transforms,
            img_phot_transforms = img_phot_transforms,
            to_tensor = to_tensor,
            ignore_encoding = ignore_encoding
        )
    
    def __len__(self) -> int:
        return len(self.img_paths)
    
    def get_raw_item(self, idx: int) -> SegSample:
        return {
            'image': Image.open(self.img_paths[idx]).convert('RGB'),
            'mask': Image.open(self.mask_paths[idx]).convert('RGB')
        } 
    
    def _setup_dataset(self) -> None:
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
    
    def _make_class_info(self) -> ClassInfo:
        class_info = {
            'human': {'idx': 1, 'rgb': (255, 255, 255)},
            'background': {'idx': 0, 'rgb': (0, 0, 0)}
        }
        return class_info