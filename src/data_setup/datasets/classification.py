#####################################
# Imports & Dependencies
#####################################
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import numpy as np
import datasets
from datasets import ClassLabel
from typing import Optional, Literal, List, Union, Callable

from src.ml_types import ImageInput, ImageLabel, SampleDict, SampleListDict, IndexLike
from src.utils import transpose_list_dict


#####################################
# Classes
#####################################
class HFClassificationDataset(Dataset):
    '''
    PyTorch Dataset wrapper around a Hugging Face classification dataset.
    This allows for optional per-sample image transforms.
    
    Args:
        hf_dataset (datasets.Dataset): Hugging Face dataset containing classification samples.
                                       Each sample must be a dictionary (SampleDict) containing the **same** keys.
                                       Required keys are:
                                           - image (ImageInput): Image sample.
                                           - label (Union[int, torch.Tensor]): Class label for the image.
        transforms (optional, Callable): Transform pipeline applied to each sample.
                                         This must be compatible with the samples in `hf_dataset`.
                                         It should accept and return a `SampleDict`.
        class_names (optional, List[str]): List of class names, where index corresponds to label encoding in `hf_dataset`.
                                           If not provided, will try to fallback to 
                                           `hf_dataset.features['label'].names` if available.
        to_tensor (bool): Whether to apply tensor and datatype (`float32`) conversions to all samples.
                          If `True`, the transform pipeline becomes:
                                1. Image tensor conversion using `ToImage()`
                                2. Optional user-provided transforms from `transforms`
                                3. Datatype conversion using `ToDtype(torch.float32, scale = True)`
                          Default is `True`.
        
    Note: The full transform pipeline can be accessed through `self.transform_pipeline`.
    '''
    def __init__(
        self, 
        hf_dataset: datasets.Dataset, 
        transforms: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
        to_tensor: bool = True
    ):
        self.hf_dataset = hf_dataset

        # Create transform pipeline
        self._init_transform_pipeline(transforms, to_tensor)

        # Setup class names
        if class_names is not None:
            self.class_names = class_names
        else:
            feature_labels = hf_dataset.features.get('label')
            if isinstance(feature_labels, ClassLabel):
                self.class_names = feature_labels.names
            else:
                self.class_names = None

        if self.class_names is not None:
            self.class_to_idx = {i: label for i, label in enumerate(self.class_names)}
            self.idx_to_class = {label: i for label, i in enumerate(self.class_names)}
        else:
            self.class_to_idx = None
            self.idx_to_class = None

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __repr__(self) -> str:
        return self.hf_dataset.__repr__()
    
    def __getitem__(self, idxs: IndexLike) -> Union[SampleDict, SampleListDict]:
        '''
        Gets a single-sample or multi-sample dictionary containing image and label information.
        Images are transformed if provided and labels are converted to tensors.

        Args:
            idxs (IndexLike): Index or collection of indices for the samples to retrieve.
            
        Returns:
            A single-sample or multi-sample dictionary depending 
            whether the input was a single index or a collection of indices.

            Single-Sample (SampleDict) has the keys (non-exhaustive):
                - image (ImageInput): Transformed image sample (original if no transforms).
                - label (ImageLabel): Class label for the image.

            Multi-Sample (SampleListDict) has the keys (non-exhaustive):
                - image (List[ImageInput]): List of transformed image samples (original if no transforms).
                - label (List[ImageLabel]): List of class labels for the images.
        '''
        if isinstance(idxs, int):
            # Indexing with a single integer
            return self._get_single_item(idxs)
        elif isinstance(idxs, np.ndarray):
            idxs = idxs.tolist()
        elif isinstance(idxs, torch.Tensor):
            idxs = idxs.tolist()
        elif not isinstance(idxs, (list, tuple)):
            raise TypeError(
                'Dataset indexing only supports integers, lists, tuples, '
                f'tensors, and numpy arrays (all integers). Got {type(idxs)}'
        )

        # Indexing with multiple integers
        items = [self._get_single_item(idx) for idx in idxs]
        return transpose_list_dict(items, mode = 'to_cols')
    
    def _get_single_item(self, idx: int) -> SampleDict:
        '''
        Gets a sample dictionary containing image and label information,
        given a **singe** index.

        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            SampleDict: Dictionary containing:
                - image (ImageInput): Transformed image sample (original if no transforms).
                - label (ImageLabel): Class label for the image.
        '''
        if not isinstance(idx, int):
            raise TypeError(f'Expected integer index. Got: {type(idx)}')
        
        item = self.hf_dataset[idx].copy()

        if self.transform_pipeline is not None:
            item = self.transform_pipeline(item)

        if self.to_tensor:
            label = item['label']
            if not isinstance(label, torch.Tensor):
                item['label'] = torch.tensor(label, dtype = torch.long)
            else:
                item['label'] = label.to(torch.long)

        return item

    def _make_transform_pipeline(self) -> None:
        '''
        Creates a single torchvision `v2.Compose` pipeline that may include:
            1. Image tensor conversion (if `self.to_tensor = True`)
            2. User-provided transforms (if `self._transforms is not None`)
            3. Float32 conversion and scaling (if `self.to_tensor = True`)

        The final `v2.Compose` pipeline is stored in `self.transform_pipeline`.
        If `self.return_tensor = False` and `self._transforms is None`, 
        then `self.transform_pipeline` will be `None.
        '''
        pipeline = []
        if self._to_tensor:
            pipeline.append(v2.ToImage()) # Converts to an image tensor

        if self._transforms is not None:
            pipeline.append(self._transforms)

        if self._to_tensor:
            pipeline.append(v2.ToDtype(torch.float32, scale = True)) # Converts to float32 and scale

        if len(pipeline) > 0:
            self.transform_pipeline = v2.Compose(pipeline)
        else:
            self.transform_pipeline = None

    def _init_transform_pipeline(self, transforms: Callable, to_tensor: bool) -> None:
        '''
        Initializes the transform pipeline `self.transform_pipeline`.
        Also sets the internal attributes `self._transforms` and `self._to_tensor`.
        '''
        self._transforms = transforms
        self._to_tensor = to_tensor
        self._make_transform_pipeline()

    @property
    def transforms(self) -> Optional[Callable]:
        '''
        Returns user-provided transforms.
        '''
        return self._transforms

    @transforms.setter
    def transforms(self, value) -> None:
        '''
        Sets user-provided transforms.
        Rebuilds transform pipeline when changed.
        '''
        self._transforms = value
        self._make_transform_pipeline()

    @property
    def to_tensor(self) -> bool:
        '''
        Returns boolean for whether Image tensor conversions are applied.
        '''
        return self._to_tensor
    
    @to_tensor.setter
    def return_tensor(self, value) -> None:
        '''
        Sets whether Image tensor conversions are applied.
        Rebuilds transform pipeline when changed.
        '''
        self._to_tensor = value
        self._make_transform_pipeline()


class HumanBinaryDataset(HFClassificationDataset):
    '''
    Human vs non-human binary classification dataset from 
    https://huggingface.co/datasets/prithivMLmods/Human-vs-NonHuman

    Note that the raw Hugging Face (HF) dataset does not have a training/validation split,
    so it is created using the built-in `train_test_split()` method and a split seed.
    Additionally, a letterbox transform (fill = 0) has already been applied to the images in the raw HF dataset.

    A sample from the raw human vs non-human HF dataset is a dictionary with the keys:
        - image (PIL.Image): 224 x 244 image in RGB format.
        - label (int): Class index for the image sample.

    HumanBinaryDataset will optionally transform the raw samples and also optionally convert them to tensors.

    Args:
        split (Literal['train', 'val']): Whether to construct the training dataset (`train`) 
                                         or the validation dataset (`val`).
        transforms (optional, Callable): Transform pipeline applied to each sample.
                                         This must be compatible with the samples of the human vs non-human dataset.
                                         It should accept and return a `SampleDict`.
        to_tensor (bool): Whether to apply tensor and datatype (`float32`) conversions to all samples.
                          If `True`, the transform pipeline becomes:
                                1. Image tensor conversion using `ToImage()`
                                2. Optional user-provided transforms from `transforms`
                                3. Datatype conversion using `ToDtype(torch.float32, scale = True)`
                          Default is `True`.
        train_frac (float): The fraction of the raw dataset to use as the training dataset.
                            The remaining fraction (`1 - train_frac`) is used as the validation dataset.
                            Default is `0.9` (90% of raw data for training, 10% of raw data for validation).
        split_seed (int): The random seed used to split the raw dataset into training/validation components.
                          Changing this will change what data is in the training/validation datasets.
                          Default is `0`.
    '''
    def __init__(
        self, 
        split: Literal['train', 'val'], 
        transforms: Optional[Callable] = None,
        to_tensor: bool = True,
        train_frac: float = 0.9, 
        split_seed: int = 0
    ):
        self.split = split
        self.train_frac = train_frac
        
        # Load raw HF dataset
        raw_dataset = datasets.load_dataset('prithivMLmods/Human-vs-NonHuman', split = 'train')
        
        # Create train/val split
        dataset_split = raw_dataset.train_test_split(test_size = (1 - train_frac), seed = split_seed)
        dataset_split_key = 'test' if split == 'val' else split # train -> train and val -> test
        
        # Class names
        # class_names = ['human', 'nonhuman']
        
        # Initialize HFClassificationDataset
        super().__init__(
            hf_dataset = dataset_split[dataset_split_key],
            transforms = transforms,
            to_tensor = to_tensor
        )