#####################################
# Imports & Dependencies
#####################################
import pandas
from PIL import Image
from pathlib import Path

import zipfile
import shutil
import subprocess

import warnings
from typing import Union, Optional, Literal, Callable, Sequence

from src.data_setup.datasets.segmentation import ClassSpec, SegmentationDatasetBase
from src.data_setup.types import SegSample


#####################################
# Dataset Class
#####################################
class SuperviselyPersonFiltered(SegmentationDatasetBase):
    '''
    Implements a dataset class for the filtered Supervisely Person dataset from
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
            The dataset will be stored in `root/supervisely_person_filtered`.
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
        self.data_dir = self.root / 'supervisely_person_filtered'
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
        Gets a single-sample dictionary from the filtered Supervisely Person dataset.
        All images and masks are converted to RGB format.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            SegSample:
                Single-sample dictionary containing (non-exhaustive):
                    - image (ImageInput): Image sample from the filtered Supervisely Person dataset.
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
        Downloads and prepares the filtered Supervisely Person dataset.

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
                f'A supervisely_person_filtered directory already exists at the root {self.root}. '
                'The filtered Supervisely Person dataset will not be downloaded.',
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