#####################################
# Imports & Dependencies
#####################################
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import requests
from abc import ABC, abstractmethod

import datasets
from datasets import ClassLabel
from pathlib import Path
from typing import Optional, Literal, List, Union, Callable, Sequence

from src.utils.shape_utils import _validate_ndim
from src.utils.data_utils import transpose_list_dict, format_idxs, all_or_none
from src.ml_types import ImageInput, ImageLabel, IndexLike
from src.data_setup.types import ClsSample, ClsSampleList


#####################################
# Base Classification Class
#####################################
class ClassificationDatasetBase(ABC, Dataset):
    '''
    Base class for image classification datasets.

    Dataset Notes: 
        - See `sec.data_setup.types` for details on 
          the sample dictionary types: `ClsSample` and `ClsSampleList`.

    Tensor and Transform Notes:
        - If images are converted to tensors (from `transforms` or `to_tensor`),
          they are expected to be 3D (channels, height, width).

        - `v2.Normalize` only acts on tensors.
          If normalization is enabled (`norm_mean` and `norm_std` are provided),
          ensure that either `to_tensor=True` or the provided `transforms` output a tensor.

    Args:
        transforms (optional, Callable): 
            Transform pipeline applied to each image.
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
    '''
    def __init__(
        self, 
        transforms: Optional[Callable] = None, 
        to_tensor: bool = True,
        norm_mean: Optional[Sequence[float]] = None,
        norm_std: Optional[Sequence[float]] = None
    ):
        if not all_or_none(norm_mean, norm_std):
            raise ValueError(
                'norm_mean and norm_std must either both be provided or both be None (not provided).'
            )
        
        self.transforms = transforms
        self.to_tensor = to_tensor
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        # Create transform pipeline
        self._make_transform_pipeline()

    def __getitem__(self, idxs: IndexLike) -> Union[ClsSample, ClsSampleList]:
        '''
        Gets a single-sample or multi-sample dictionary containing image and label data.
        If a transform pipeline is available, the images and labels are transformed accordingly 
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
            Union[ClsSample, ClsSampleList]:
                A single-sample or multi-sample dictionary depending on
                whether the input was a single index or a collection of indices.

                Single-Sample (ClsSample) contains the following (non-exhaustive):
                    - image (ImageInput): Transformed image sample (original if no transform pipeline).
                    - label (ImageLabel): Class label for the image.

                Multi-Sample (ClsSampleList) contains the following (non-exhaustive):
                    - image (List[ImageInput]): List of transformed image samples (original if no transform pipeline).
                    - label (List[ImageLabel]): List of class labels for the images.
        '''
        if isinstance(idxs, int):
            # Indexing with a single integer
            return self.get_single_item(idxs)
        else:
            # Indexing with multiple integers
            idxs = format_idxs(idxs)
            items = [self.get_single_item(idx) for idx in idxs]
            return transpose_list_dict(items, mode = 'to_cols')
    
    def get_single_item(self, idx: int) -> ClsSample:
        '''
        Gets a sample dictionary containing image and label data, given a **single** index.
        If a transform pipeline is available, the image and label is transformed accordingly 
        (see `_make_transform_pipeline` for details).

        Args:
            idx (int): 
                Index of the sample to retrieve.
            
        Returns:
            ClsSample: 
                Single-sample dictionary containing (non-exhaustive):
                    - image (ImageInput): Transformed image sample (original if not transform pipeline).
                    - label (ImageLabel): Class label for the image.
        '''
        if not isinstance(idx, int):
            raise TypeError(f'Expected integer index. Got: {type(idx)}')
        
        item = self.get_base_item(idx)
        
        if self.transform_pipeline is not None:
            # Image in item is always a tensor if self.to_tensor = True
            item = self.transform_pipeline(item)
            
        img, label = item['image'], item['label']
        if isinstance(img, torch.Tensor):
            # Validate image tensor of shape (channels, height, width)
            _validate_ndim(img, ndim = 3, context_name = 'images')

        # Convert labels to integer tensors if needed
        if isinstance(label, torch.Tensor):
            item['label'] = label.to(torch.long)
        elif self.to_tensor:
            item['label'] = torch.tensor(label, dtype = torch.long)
        
        return item

    def _make_transform_pipeline(self) -> None:
        '''
        Creates a single torchvision transform pipeline that may include:
            1. User-provided transforms (if `transforms` is provided)
            2. Image tensor conversion (if `to_tensor=True`)
            3. Float32 conversion and scaling (if `to_tensor=True`)
            4. Normalization (if `norm_mean` and `norm_std` are provided)

        The final transform pipeline can be accessed through the attribute `transform_pipeline`.
        There will be no transform pipeline (`transform_pipeline` is `None`), 
        if all these conditions are met:
            - `transforms` is not provided
            - `to_tensor=False`
            - `norm_mean` and `norm_std` are not provided (both `None`)
        '''
        pipeline = []
        if self.transforms is not None:
            pipeline.append(self.transforms)

        if self.to_tensor:
            # Note: I put ToImage() after user-provided transforms
                # because geometric transforms seem to work better with PIL images
            pipeline.extend([
                v2.ToImage(), # Converts to an image tensor
                v2.ToDtype(torch.float32, scale = True) # Converts to float32 and scale
            ])

        if self.norm_mean is not None:
            pipeline.append(
                v2.Normalize(mean = self.norm_mean, std = self.norm_std)
            )

        self.transform_pipeline = v2.Compose(pipeline) if (len(pipeline) > 0) else None

    @abstractmethod
    def __len__(self) -> int:
        '''
        Returns the number of samples in the dataset
        '''
        pass

    @abstractmethod
    def get_base_item(self, idx: int) -> ClsSample:
        '''
        Gets a base dictionary containing information for a single sample
        prior to applying the dataset transform pipeline.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            ClsSample:
                Single-sample dictionary containing (non-exhaustive):
                    - image (ImageInput): Base image sample.
                    - label (ImageLabel): Base class label for the image.
        '''
        pass


#####################################
# Hugging Face Dataset Classes
#####################################
class HFClassificationDataset(ClassificationDatasetBase):
    '''
    `ClassificationDatasetBase` wrapper around a Hugging Face classification dataset.
    This allows for optional per-sample image transforms.
    
    Tensor and Transform Notes:
        - `v2.Normalize` only acts on tensors.
          If normalization is enabled (`norm_mean` and `norm_std` are provided),
          ensure that either `to_tensor=True` or the provided `transforms` output a tensor.
          
        - If `transforms` are provided, it is recommended that the Hugging Face dataset (`hf_dataset`)
          does not have transforms attached to it 
          (either through `hf_dataset.set_transform()` or `hf_dataset.with_transform()`).
          This ensures that there is only a single source of transforms.

    Args:
        hf_dataset (datasets.Dataset): 
            Hugging Face dataset containing classification samples.
            Each sample must be a dictionary (ClsSample) containing the same keys.
            Required fields:
                - image (ImageInput): Image sample.
                - label (Union[int, torch.Tensor]): Class label for the image.
        transforms (optional, Callable): 
            Transform pipeline applied to each image.
            This must be compatible with the samples in `hf_dataset`.
            This must accept and return a sample dictionary (`ClsSample`).
        class_names (optional, List[str]): 
            List of class names, where index corresponds to label encoding in `hf_dataset`.
            If not provided, will try to fallback to `hf_dataset.features['label'].names` if available.
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
    '''
    def __init__(
        self, 
        hf_dataset: datasets.Dataset, 
        transforms: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
        to_tensor: bool = True,
        norm_mean: Optional[Sequence[float]] = None,
        norm_std: Optional[Sequence[float]] = None
    ):
        super().__init__(
            transforms = transforms, 
            to_tensor = to_tensor, 
            norm_mean = norm_mean, 
            norm_std = norm_std
        )
        self.hf_dataset = hf_dataset

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
            self.class_to_idx, self.idx_to_class = {}, {}
            for i, label in enumerate(self.class_names):
                self.class_to_idx[label] = i
                self.idx_to_class[i] = label
        else:
            self.class_to_idx, self.idx_to_class = None, None

    def __len__(self) -> int:
        '''
        Returns the number of samples in the HF dataset.
        '''
        return len(self.hf_dataset)

    def __repr__(self) -> str:
        '''
        Returns a representative string for the HF dataset.
        '''
        return self.hf_dataset.__repr__()
    
    def get_base_item(self, idx: int) -> ClsSample:
        '''
        Gets a single-sample dictionary directly from the HF dataset.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            ClsSample:
                Single-sample dictionary from `hf_dataset` containing:
                    - image (ImageInput): Image sample.
                    - label (ImageLabel): Class label for the image.
        '''
        return self.hf_dataset[idx].copy()
    

class MiniImageNetDataset(HFClassificationDataset):
    '''
    Wraps the Mini-ImageNet HF dataset from https://huggingface.co/datasets/timm/mini-imagenet.

    This applies the following to the HF dataset samples:
        - Optional user-provided transforms
        - Optional tensor and datatype conversion
        - Optional normalization

    Dataset Notes:
        - A sample from the Mini-ImageNet HF dataset is a dictionary containing:
            - image (Image.Image): PIL image with shape varying per sample.
            - label (int): Class index for the image sample.

    Tensor and Transform Notes:
        - `v2.Normalize` only acts on tensors.
          If normalization is enabled (`norm_mean` and `norm_std` are provided),
          ensure that either `to_tensor=True` or the provided `transforms` output a tensor.

    Args:
        root (Union[str, Path]): 
            Directory to download the dataset in. This is the `cache_dir` of the HF dataset.
        split (Literal['train', 'val', 'test']): 
            Whether to construct the training dataset (`train`), 
            validation dataset (`val`), or test dataset (`test`).
        transforms (optional, Callable): 
            Transform pipeline applied to each image.
            This must be compatible with the samples of the Mini-ImageNet dataset.
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
    '''
    def __init__(
        self, 
        root: Union[str, Path],
        split: Literal['train', 'val', 'test'],
        transforms: Optional[Callable] = None,
        to_tensor: bool = True,
        norm_mean: Optional[Sequence[float]] = None,
        norm_std: Optional[Sequence[float]] = None
    ):
        self.split = split
        self.root = Path(root)
        self.data_dir = self.root / 'timm___mini-imagenet'

        # Load Mini-ImageNet HF dataset
        self.hf_dataset = datasets.load_dataset(
            'timm/mini-imagenet',
            cache_dir = root,
            split = split if split != 'val' else 'validation'
        )

        class_names = self._get_class_names()

        # Initialize HFClassificationDataset
        super().__init__(
            hf_dataset = self.hf_dataset,
            class_names = class_names,
            transforms = transforms,
            to_tensor = to_tensor,
            norm_mean = norm_mean,
            norm_std = norm_std
        )

    def get_base_item(self, idx: int) -> ClsSample:
        '''
        Gets a single-sample dictionary from the Mini-ImageNet HF dataset.
        All PIL images are converted to RGB format.

        Args:
            idx (int):
                Index of the sample to retrieve.

        Returns:
            ClsSample:
                Single-sample dictionary containing:
                    - image (Image.Image): PIL image from the Mini-ImageNet HF dataset,
                                           converted to RGB format.
                    - label (int): Class label for from the Mini-ImageNet HF dataset.
        '''
        item = self.hf_dataset[idx].copy()
        item['image'] = item['image'].convert('RGB') # Convert all to RGB (3 channels)
        return item

    def _get_class_names(self) -> List[str]:
        '''
        Returns a list of class names for the Mini-ImageNet HF dataset
        The ordering of the list matches the index order of the dataset.
        '''
        # Fetch synset ID to class name mapping
        mapping_url = 'https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/raw'
        response = requests.get(mapping_url)
        response.raise_for_status()

        synset_to_class = {}
        for line in response.text.splitlines():
            labeling = line.strip().split()
            synset_to_class[labeling[0]] = labeling[2]

        self.synset_to_class = synset_to_class
        
        # Get class names from synset ids
        self.synset_ids = self.hf_dataset.features['label'].names
        return [synset_to_class[synset_id] for synset_id in self.synset_ids]


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