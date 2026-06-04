#####################################
# Imports & Dependencies
#####################################
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SequentialSampler, RandomSampler

import warnings
from typing import List, Union, Optional, Literal, Literal, Callable

from src.ml_types import SpatialSize
from src.utils import all_or_none
from src.data_setup.multiscale import MultiScaleBatchSampler, MultiScaleWrapper


#####################################
# Functions
#####################################
def build_dataloader(
    dataset: Dataset,
    split: Literal['train', 'val', 'test'],
    batch_size: int,
    num_workers: int = 0,
    prefetch_factor: Optional[int] = None,
    collate_fn: Optional[Callable] = None,
    drop_last: bool = False,
    multiscale_interval: Optional[int] = None,
    multiscale_sizes: Optional[List[SpatialSize]] = None,
    resize_fn: Optional[Callable] = None,
    device: Union[torch.device, str] = 'cpu',
    **resize_kwargs
) -> DataLoader:
    '''
    Create a training, validation, or testing dataloader for a dataset.
    This supports optional multiscale training, 
    which changes the spatial size of batches every few intervals.

    Note: To enable multiscale training, 
          `multiscale_interval`, `multiscale_sizes`, and `resize_fn` must all be provided.
          Otherwise, all three arguments must be `None` (not provided).
    
    Args:
        dataset (Dataset): 
            The dataset to create the dataloader for.
        split (Literal['train', 'val', 'test']): 
            The dataset split. 
            This Controls the shuffling behavior:
                - train: `shuffle = True`
                - val or test: `shuffle = False`
        batch_size (int): 
            Number of samples per batch.
        num_workers (int): 
            Number of workers for multiprocessing. 
            Default is `0` (no multiprocessing).
        prefetch_factor (optional, int): 
            Number of batches pre-loaded in advance per worker.
            If `num_workers = 0`, this is forced to `None`.
        collate_fn (optional, Callable): 
            Collate function to merge samples into batches.
        drop_last (bool): 
            Whether to drop the last remaining samples in dataset if 
            they cannot create a full batch of size batch_size.
            Default is `False`.
        multiscale_interval (optional, int):
            Batch interval to change input size for multiscale training.
        multiscale_sizes (optional, List[SpatialSize]):
            List of input sizes to use during multiscale training.
            Each elements can be `int` (assumed square) or `(height, width)` tuples.
        resize_fn (optional, Callable):
            The resize function to use when resizing images for multiscale training.
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
        device (Union[torch.device, str]): 
            Device to send batched tensors to. Default is `cpu`.
        **resize_kwargs:
            Keyword arguments apart from `item` and `size` to use when calling `resize_fn`.

    Returns:
        DataLoader: 
            The dataloader for the dataset.
    '''
    if not all_or_none(multiscale_interval, multiscale_sizes, resize_fn):
        raise ValueError(
            'multiscale_interval, multiscale_sizes, and resize_fn must either all be provided or all be None.'
        )
    
    # Device related parameters
    device = torch.device(device)
    if device.type == 'cuda':
        mp_context = None
        pin_memory = True
    elif device.type == 'mps':
        mp_context = 'forkserver'
        pin_memory = False
    else:
        mp_context = None
        pin_memory = False

    if num_workers > 0:
        persistent_workers = True
    else:
        if prefetch_factor is not None:
            warnings.warn(
                'prefetch_factor is ignored when num_workers = 0; setting it to None.',
                UserWarning
            )
        prefetch_factor = None
        mp_context = None
        persistent_workers = False
    
    loader_kwargs = {
        'num_workers': num_workers,
        'prefetch_factor': prefetch_factor,
        'collate_fn': collate_fn,
        'multiprocessing_context': mp_context,
        'pin_memory': pin_memory,
        'persistent_workers': persistent_workers
    }
    
    # Sampler
    if split == 'train':
        sampler = RandomSampler(dataset) # Shuffles every epoch
    else:
        sampler = SequentialSampler(dataset)
        
    # Multiscale training
    if multiscale_interval is None:
        loader_kwargs.update({
            'dataset': dataset,
            'batch_size': batch_size,
            'sampler': sampler,
            'drop_last': drop_last
        })
    else:
        # Wrap dataset to enable multiscale resizing within __getitem__
        loader_kwargs['dataset'] = MultiScaleWrapper(
            dataset = dataset,
            resize_fn = resize_fn,
            **resize_kwargs
        )
        
        # Batch sampler returns lists of (idx, size)
        loader_kwargs['batch_sampler'] = MultiScaleBatchSampler(
            sampler = sampler,
            batch_size = batch_size,
            multiscale_interval = multiscale_interval,
            multiscale_sizes = multiscale_sizes,
            drop_last = drop_last
        ) 
        
    return DataLoader(**loader_kwargs)