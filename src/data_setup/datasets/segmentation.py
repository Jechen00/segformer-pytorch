#####################################
# Imports & Dependencies
#####################################
import torch
from torchvision.transforms import v2

import pandas
from PIL import Image
from pathlib import Path

import zipfile
import shutil
import subprocess

import warnings
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Literal, TypedDict, TypeAlias

from src.data_setup.transforms.ops import ImageTransform, ToImageAndMask
from src.utils import make_tuple, transpose_list_dict, normalize_idxs
from src.masks import rgb_to_idx_mask
from src.ml_types import RGBTuple, IndexLike
from src.data_setup.types import SegSample, SegSampleList


#####################################
# Class Typings
#####################################
class ClassSpec(TypedDict):
    idx: int
    clr: RGBTuple

ClassInfo: TypeAlias = Dict[str, ClassSpec]


#####################################
# Base Segmentation Class
#####################################
class SegmentationDatasetBase(ABC):
    def __init__(
        self,
        class_info: ClassInfo,
        geo_transforms: Optional[v2.Compose] = None,
        img_phot_transforms: Optional[v2.Compose] = None,
        to_tensor: bool = True,
        ignore_encoding: Optional[ClassSpec] = None
    ):
        self.class_info = class_info
        self.ignore_encoding = ignore_encoding
        self.geo_transforms = geo_transforms
        self.img_phot_transforms = img_phot_transforms
        self.to_tensor = to_tensor
        
        # Class name, index, and color mappings
        self._make_mappings() # Creates class_to_idx, idx_to_class, clr_to_idx, idx_to_clr

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
                                - A numpy array of integers
                                - A tensor of integers
        '''
        if isinstance(idxs, int):
            # Indexing with a single integer
            return self.get_single_item(idxs)
        else:
            # Indexing with multiple integers
            idxs = normalize_idxs(idxs)
            items = [self.get_single_item(idx) for idx in idxs]
            return transpose_list_dict(items, mode = 'to_cols')
        
    def get_single_item(self, idx: int) -> SegSample:        
        item = self.get_raw_item(idx)
        
        if self.transform_pipeline is not None:
            # Image and mask in item are always tensors if self.to_tensor = True
            item = self.transform_pipeline(item)
            
        if self.to_tensor:
            # Normalize mask to index mask (training-ready)
            item['mask'] = self._normalize_mask(item['mask'])
            return item
        else:
            return item
            
    def _make_mappings(self) -> None:
        # Create mappings based on index
        self.idx_to_class, self.class_to_idx = {}, {}
        self.idx_to_clr, self.clr_to_idx = {}, {}
        
        for name, info in self.class_info.items():
            clr, idx = info['clr'], info['idx']
            
            # Check for duplicates
            if clr in self.clr_to_idx:
                raise ValueError(f"Duplicate color mapping detected for color ('clr'): {clr}")
            if idx in self.idx_to_clr:
                raise ValueError(f"Duplicate index mapping detected for index ('idx'): {idx}")
                
            self.idx_to_class[idx] = name
            self.class_to_idx[name] = idx
            self.idx_to_clr[idx] = clr
            self.clr_to_idx[clr] = idx

        # Optionally add ignore index to clr_to_idx, idx_to_clr
        self._add_ignore_mapping()
            
    def _add_ignore_mapping(self) -> None:
        ignore_encoding = self.ignore_encoding
        if self.ignore_encoding is None:
            self.ignore_idx, self.ignore_clr = None, None
            self._unmapped_idx = -1000 # Unmapped RGB pixels will be flagged ad errors
            return

        idx, clr = ignore_encoding['idx'], ignore_encoding['clr']
        clr = make_tuple(clr, 3) # Normalize color

        # Check for duplicates
        if clr in self.clr_to_idx:
            if idx != self.clr_to_idx[clr]:
                raise ValueError(f"Ignore color conflicts with the class color ('clr'): {clr}")
        if idx in self.idx_to_clr:
            if clr != self.idx_to_clr[idx]:
                raise ValueError(f"Ignore index conflicts with the class index ('idx'): {idx}")

        self.ignore_idx, self.ignore_clr = idx, clr
        self.clr_to_idx[clr] = idx
        self.idx_to_clr[idx] = clr

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

    def _normalize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        '''
        Normalizes a segmentation mask to an index mask.

        If `mask` is already an index mask (2D tensor), return it unchanged.
        If `mask` is a RGB mask (3D tensor) to an index mask.
        if `mask` is not a 2D or 3D tensor, throw an error.
        '''
        mask_ndim = mask.ndim
        if mask_ndim == 2:
            # Mask is already an index mask
            return mask # Shape: (height, width)
        
        elif mask_ndim == 3:
            # Mask is a RGB mask --> need to convert to index mask
            idx_mask = rgb_to_idx_mask(
                rgb_mask = mask,
                rgb_to_idx = self.clr_to_idx,
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
        data_dir: Union[str, Path], 
        split: Literal['train', 'val'],
        geo_transforms: Optional[v2.Compose] = None,
        img_phot_transforms: Optional[v2.Compose] = None,
        to_tensor: bool = True,
        ignore_encoding: Optional[ClassSpec] = None,
        train_frac: float = 0.8,
        split_seed: int = 0
    ):
        self.data_dir = Path(data_dir)
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
                f'A directory at {data_dir} already exists. '
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
            'human': {'idx': 1, 'clr': (255, 255, 255)},
            'background': {'idx': 0, 'clr': (0, 0, 0)}
        }
        return class_info