#####################################
# Imports & Dependencies
#####################################
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import numpy as np
import datasets
from typing import Optional, Literal, List, Tuple, Union, TypeAlias

from src.ml_types import ImageInput, SampleDict, SampleListDict
from src.utils import transpose_list_dict

IndexLike: TypeAlias = Union[int, List[int], Tuple[int, ...], torch.Tensor, np.ndarray]


#####################################
# Classes
#####################################
class HFClassificationDataset(Dataset):
    '''
    PyTorch Dataset wrapper around a Hugging Face classification dataset.
    This allows for optional per-sample image transformations.
    
    Args:
        hf_dataset (datasets.Dataset): Hugging Face dataset containing classification samples.
                                       Each sample must be a dictionary (SampleDict) containing the **same** keys.
                                       Required keys are:
                                           - image (ImageInput): Image sample.
                                           - label (Union[int, torch.Tensor]): Class label for the image.
        aug_transforms (optional, v2.Compose): Data augmentation pipeline applied to each image independently.
                                               This must be compatible with the samples in `hf_dataset`
                                               (e.g. accepts a SampleDict).
        base_transforms (optional, v2.Compose): Base transform pipeline separate from augmentation transforms.
                                                This must be compatible with the samples in `hf_dataset`
                                                (e.g. accepts a SampleDict).
        base_first (bool): Whether to apply `aug_transforms` first (if provided) or `base_transforms` first (if provided).
                           Default is `False` (`aug_transforms` are applied first).
    '''
    def __init__(
        self, 
        hf_dataset: datasets.Dataset, 
        aug_transforms: Optional[v2.Compose] = None,
        base_transforms: Optional[v2.Compose] = None,
        base_first: bool = False
    ):
        self.hf_dataset = hf_dataset
        self.aug_transforms = aug_transforms
        self.base_transforms = base_transforms
        self.base_first = base_first

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
                - label (torch.tensor): Class label for the image.

            Multi-Sample (SampleListDict) has the keys (non-exhaustive):
                - image (List[ImageInput]): List of transformed image samples (original if no transforms).
                - label (List[torch.tensor]): List of class labels for the images.
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
                - label (torch.tensor): Class label for the image.
        '''
        if not isinstance(idx, int):
            raise TypeError(f'Expected integer index. Got: {type(idx)}')
        
        item = self.hf_dataset[idx].copy()
        transforms = [self.aug_transforms, self.base_transforms]
        if self.base_first:
            transforms = transforms[::-1]

        for transform in transforms:
            if transform is not None:
                item = transform(item)

        label = item['label']
        if not isinstance(label, torch.Tensor):
            item['label'] = torch.tensor(label, dtype = torch.long)

        return item


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

    HumanBinaryDataset will optionally transform the raw samples and always converts the labels to tensors.

    Args:
        split (Literal['train', 'val']): Whether to construct the training dataset (`train`) or the validation dataset (`val`).
        aug_transforms (optional, v2.Compose): Data augmentation pipeline applied to each image independently.
                                               This need to be compatible with the samples of 
                                               the human vs non-human dataset 
                                               (i.e. accepts the dictionary input with `image` and `label` keys).
        base_transforms (optional, v2.Compose): Base transform pipeline separate from augmentation transforms.
                                                This need to be compatible with the samples of 
                                                the human vs non-human dataset 
                                                (i.e. accepts the dictionary input with `image` and `label` keys).
        base_first (bool): Whether to apply `aug_transforms` first (if provided) or `base_transforms` first (if provided).
                           Default is `False` (`aug_transforms` are applied first).
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
        aug_transforms: Optional[v2.Compose] = None,
        base_transforms: Optional[v2.Compose] = None,
        base_first: bool = False,
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
        
        # Class names and indices
        self.class_names = ['human', 'nonhuman']
        self.class_to_index = {i: label for i, label in enumerate(self.class_names)}
        
        # Initialize HFClassificationDataset
        super().__init__(
            hf_dataset = dataset_split[dataset_split_key],
            aug_transforms = aug_transforms,
            base_transforms = base_transforms,
            base_first = base_first
        )