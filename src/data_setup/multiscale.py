#####################################
# Imports & Dependencies
#####################################
from torch.utils.data import Dataset, Sampler

import random, math
from typing import (
    List, Union, Tuple, Optional,
    Callable, Iterable, Iterator
)

from src.ml_types import SpatialSize, SampleDict, ImageInput
from src.utils import make_tuple


#####################################
# Classes
#####################################
class MultiScaleWrapper():
    '''
    Wrapper used to enable multiscale training for a base dataset.

    Expects the `__getitem__` method of the base dataset to accept only an integer index.
    The wrapper modifies `__getitem__` to accepts both integer indices and tuples of the form `(index, input_size)`,
    where `index` is the image index and `input_size` is the size used for resizing in multiscale training.

    Args:
        dataset (Dataset): The base dataset to modify.
        resize_fn (Callable): The resize function to use when resizing images.
                              This function must accept:
                                - item (SampleDict): 
                                        Dictionary containing the item information.
                                        This always includes:
                                            - image (ImageInput): image sample.
                                        It may also contain other keys depending on the task.
                                        For example:
                                            - label (Union[int, torch.Tensor]): class index for image classification tasks.
                                            - mask (ImageInput): segmentation mask for image segmentation tasks.
                                - size (SpatialSize): Size (height, width) used to resize `item['image']`.
                                                      If `int`, size is assumed to be square.
        default_size (optional, SpatialSize): A default size to use when `__getitem__` is called 
                                              with only an integer index. 
                                              If `None`, the default is that no resizing is applied.
        resize_kwargs: Keyword arguments apart from `item` and `input_size` to use when calling `resize_fn`.
    '''
    def __init__(
        self,
        dataset: Dataset,
        resize_fn: Callable,
        default_size: Optional[SpatialSize] = None,
        **resize_kwargs
    ):
        if 'size' in resize_kwargs:
            raise ValueError('size must not be a key in resize_kwargs.')
        self.dataset = dataset
        self.resize_fn = resize_fn
        self.resize_kwargs = resize_kwargs
        self.default_size = default_size

    def __getitem__(self, input_info: Union[int, Tuple[int, SpatialSize]]) -> SampleDict:
        '''
        Gets a sample dictionary containing image and label information, given an index.
        Also optionally resizes the image if a size is provided.

        Args:
            input_info (Union[int, Tuple[int, SpatialSize]]): 
                An image index (`int`) or a `tuple` of the form `(index, input_size)`.
                If only an image index is provided, the returned image is resized using `default_size`.
                If `(index, input_size)` is provided, the returned image is resized to `input_size`.

        Returns:
            SampleDict: Dictionary containing sample information.
                        The keys include (non-exhaustive):
                            - image (ImageInput): The resized image sample.
        '''
        if isinstance(input_info, int):
            idx, size = input_info, self.default_size
        else:
            idx, size = input_info

        item = self.dataset[idx] # Get dataset items
        if size is not None:
            item = self.resize_fn(item, size, **self.resize_kwargs)
        return item


class MultiScaleBatchSampler():
    '''
    Batch sampler for multi-scale training.
    At each iteration, this sampler yields a list of `(samp_idx, input_size)` pairs,
    where input_size may change every multiscale_interval batches if enabled.

    Note: The dataset in the dataloader must support receiving `(samp_idx, input_size)` in `__getitem__`.
          This can be done by wrapping the dataset with `MultiScaleWrapper`.

    Adapted from: https://github.com/CaoWGG/multi-scale-training/blob/master/batch_sampler.py

    Args:
        sampler (Union[Sampler[int], Iterable[int]]): Base sampler (e.g., RandomSampler or SequentialSampler)
                                                      used to sample image indices.
        batch_size (int): Number of sample images per batch.
        multiscale_interval (int): Batch interval to change input size for multiscale training.
        multiscale_sizes (List[SpatialSize]): List of input sizes to use during multiscale training.
                                              Elements can be `int` (assumed square) or `(height, width)` tuples.
        drop_last (bool): Whether to drop the last remaining samples in a dataset if 
                          they cannot create a full batch of size `batch_size`.
                          If `False`, the final batch may have `size <= batch_size`,
                          increasing the total number of batches by at most one.
                          Default is `False`.
    '''
    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        multiscale_interval: int,
        multiscale_sizes: List[SpatialSize],
        drop_last: bool = False
    ):      
        self.sampler = sampler
        self.batch_size = batch_size
        self.multiscale_interval = multiscale_interval
        self.multiscale_sizes = [make_tuple(size) for size in multiscale_sizes]
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[Tuple[int, SpatialSize]]]:
        '''
        Creates an iterator that yields batches of `(samp_idx, input_size)` pairs.
        Provides multiscale training by changing `input_size` every `multiscale_interval` batches.

        Yields:
            batch (List[Tuple[int, SpatialSize]]): 
                A list of `(samp_idx, input_size)` pairs.
                    - samp_idx (int): Index of the image sample in the dataset.
                    - input_size (SpatialSize): A tuple `(height, width)` indicating the input size of the batch.
        '''
        batch = []
        num_batches = 0
        input_size = random.choice(self.multiscale_sizes) # Initialize input size
        for samp_idx in self.sampler:
            batch.append((samp_idx, input_size))
            
            if len(batch) == self.batch_size:
                yield batch
                num_batches += 1
                
                if (num_batches % self.multiscale_interval == 0):
                    input_size = random.choice(self.multiscale_sizes)
                
                batch = [] # Reset batch indices
                
        if (not self.drop_last) and (len(batch) > 0):
            yield batch # Yields the last batch, even if it is shorter than batch_size
            
    def __len__(self) -> int:
        '''
        Gets the number of batches in the dataset.
        '''
        if not self.drop_last:
            return math.ceil(len(self.sampler) / self.batch_size)
        else:
            return len(self.sampler) // self.batch_size