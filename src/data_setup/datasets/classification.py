#####################################
# Imports & Dependencies
#####################################
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from typing import Optional
import datasets

from src.utils.common_types import ImageInput


#####################################
# Classes
#####################################
class HFClassificationDataset(Dataset):
    '''
    PyTorch Dataset wrapper around a Hugging Face classification dataset.
    This allows for optional per-sample image transformations.
    
    Args:
        hf_dataset (datasets.Dataset): Hugging Face dataset containing classification samples.
                                       Each sample should be a dictionary only containing:
                                           - image (ImageInput): Image sample.
                                           - label (Union[int, torch.Tensor]): Class label for the image.
                                           
        aug_transforms (optional, v2.Compose): Data augmentation pipeline applied to each image independently.
                                               This should be compatible with the samples in hf_dataset 
                                               (e.g. accepts a dictionary input).
        base_transforms (optional, v2.Compose): Base transform pipeline separate from augmentation transforms.
                                                This should be compatible with the samples in hf_dataset
                                                (e.g. accepts a dictionary input).
        base_first (bool): Whether to apply aug_transforms first (if provided) or base_transforms first (if provided).
                           Default is False (aug_transforms are applied first).
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
    
    def __getitem__(self, idx: int) -> dict:
        '''
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: Dictionary containing:
                - image (ImageInput): Transformed image sample (original if no transforms).
                - label (torch.tensor): Class label for the image.
        '''
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