#####################################
# Imports & Dependencies
#####################################
import datasets
from pathlib import Path
from typing import Optional, Literal, Union, Callable, Sequence

from src.data_setup.datasets.classification import HFClassificationDataset


#####################################
# Dataset Class
#####################################
class HumanBinaryDataset(HFClassificationDataset):
    '''
    Wraps the Human-Nonhuman binary classification dataset from 
    https://huggingface.co/datasets/prithivMLmods/Human-vs-NonHuman

    This applies the following to the HF dataset samples:
        - Optional user-provided transforms
        - Optional tensor and datatype conversion
        - Optional normalization

    Dataset Notes:
        - A sample from the Human-Nonhuman HF dataset is a dictionary containing:
            - image (Image.Image): 224 x 244 image in RGB format.
            - label (int): Class index for the image sample.

        - The Human-Nonhuman HF dataset does not have a training/validation split,
          so it is created using the built-in `train_test_split()` method and a split seed.
          Unfortunately, this potentially leads to **data leakage**, since the HF dataset 
          tends to include multiple images of the same person (from different perspectives).
          
        - A letterbox transform (fill = 0) has already been applied to the images in the Human-Nonhuman HF dataset.

    Tensor and Transform Notes:
        - `v2.Normalize` only acts on tensors.
          If normalization is enabled (`norm_mean` and `norm_std` are provided),
          ensure that either `to_tensor=True` or the provided `transforms` output a tensor.

    Args:
        root (Union[str, Path]): 
            Directory to download the dataset in.
            This is the `cache_dir` of the HF dataset.
        split (Literal['train', 'val']): 
            Whether to construct the training dataset (`train`) or the validation dataset (`val`).
        transforms (optional, Callable): 
            Transform pipeline applied to each image.
            This must be compatible with the samples of the Human-Nonhuman HF dataset.
            This must accept and return a sample dictionary (`ClsSample`).
        to_tensor (bool): 
            Whether to apply tensor conversions to all samples **after** any provided `transforms`.
            Images are scaled to [0, 1] with dtype `torch.float32`.
            Labels are converted to dtype `torch.long`.
            Default is `True`.
        norm_mean (optional, Sequence[float]): 
            Sequence of means (one for each input channel)
            used to normalize images **after** tensor conversion.
            If provided, `norm_std` must also be provided.
        norm_std (optional, Sequence[float]): 
            Sequence of standard deviations (one for each input channel)
            used to normalize images **after** tensor conversion.
            If provided, `norm_mean` must also be provided.
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
        transforms: Optional[Callable] = None,
        to_tensor: bool = True,
        norm_mean: Optional[Sequence[float]] = None,
        norm_std: Optional[Sequence[float]] = None,
        train_frac: float = 0.9, 
        split_seed: int = 0
    ):
        self.split = split
        self.train_frac = train_frac
        self.root = Path(root)
        self.data_sir = self.root / 'prithivMLmods___human-vs-non_human'
        
        # Load Human-Nonhuman HF dataset
        hf_dataset = datasets.load_dataset('prithivMLmods/Human-vs-NonHuman', cache_dir = root, split = 'train')
        
        # Create train/val split
        dataset_split = hf_dataset.train_test_split(test_size = (1 - train_frac), seed = split_seed)
        dataset_split_key = 'test' if split == 'val' else split # train -> train and val -> test
        
        # Initialize HFClassificationDataset
        super().__init__(
            hf_dataset = dataset_split[dataset_split_key],
            transforms = transforms,
            to_tensor = to_tensor,
            norm_mean = norm_mean,
            norm_std = norm_std
        )