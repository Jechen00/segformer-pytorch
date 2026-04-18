#####################################
# Imports & Dependencies
#####################################
from torch.utils.data import Dataset, Sampler

import random, math
from typing import List, Union, Tuple, Iterable, Iterator, Optional, Callable

from src.utils.common_types import SpatialSize
from src.utils import misc


#####################################
# Classes
#####################################
class MultiScaleWrapper():
    '''
    Expects the __getitem__ method to recive an integer index.

    resize_fn must accept:
        (item: dict, size: SpatialSize, **resize_kwargs)
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

    def __getitem__(self, input_info: Union[int, Tuple[int, SpatialSize]]) -> dict:
        '''
        Args:
            input_info (Union[int, Tuple[int, SpatialSize]]):
            
        Returns:
            dict:
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
    At each iteration, this sampler yields a list of (samp_idx, input_size) pairs,
    where input_size may change every multiscale_interval batches if enabled.

    Note: The dataset in the dataloader must support receiving (samp_idx, input_size) in __getitem__.
          This can be done by wrapping the dataset with MultiScaleWrapper.

    Adapted from: https://github.com/CaoWGG/multi-scale-training/blob/master/batch_sampler.py

    Args:
        sampler (Union[Sampler[int], Iterable[int]]): Base sampler (e.g., RandomSampler or SequentialSampler)
                                                      used to sample image indices.
        batch_size (int): Number of sample images per batch.
        multiscale_interval (int): Batch interval to change input size for multiscale training.
        multiscale_sizes (List[SpatialSize]): List of input sizes to use during multiscale training.
                                              Elements can be ints (assumed square) or (height, width) tuples.
        drop_last (bool): Whether to drop the last remaining samples in a dataset if 
                          they cannot create a full batch of size batch_size.
                          If False, the final batch may have size <= batch_size,
                          increasing the total number of batches by at most one.
                          Default is False.
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
        self.multiscale_sizes = [misc.make_tuple(size) for size in multiscale_sizes]
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[Tuple[int, SpatialSize]]]:
        '''
        Creates an iterator that yields batches of (samp_idx, input_size) pairs.
        Provides multiscale training by changing input_size every multiscale_interval batches.

        Yields:
            batch (List[Tuple[int, SpatialSize]]): 
                A list of (samp_idx, input_size) pairs.
                    - samp_idx (int): Index of the image sample in the dataset.
                    - input_size (SpatialSize): A tuple (height, width) indicating 
                                                    the input size of the batch.
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