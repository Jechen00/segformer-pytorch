#####################################
# Imports & Dependencies
#####################################
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

from abc import ABC, abstractmethod
from typing import Optional, Union, Callable, Sequence

from src.data_setup.types import ClsSample, ClsSampleList

from src.utils.data_utils import transpose_list_dict, format_idxs, all_or_none
from src.utils.shape_utils import _validate_ndim
from src.ml_types import IndexLike


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