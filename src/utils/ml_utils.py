#####################################
# Imports & Dependencies
#####################################
import torch

import os
import random
import numpy as np 

from typing import Any

from src.ml_types import Aggregation


#####################################
# Functions
#####################################
def set_seed(seed: int = 0) -> None:
    '''
    Sets random seed and deterministic settings 
    for reproducibility across:
        - PyTorch
        - NumPy
        - Python's random module
        - CUDA
    
    Args:
        seed (int): 
            The seed value to set.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.use_deterministic_algorithms(True, warn_only = True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def get_device() -> torch.device:
    '''
    Returns the best available computation device.
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def recursive_to_cpu(x: Any) -> Any:
    '''
    Recursively moves all tensors in a nested dictionary/list structure to the CPU.
    Other objects (e.g. floats, ints, ndarrays) remain unchanged.
    
    Args:
        x (Any): 
            Any Python object. Tensors are moved to CPU.
            Dictionaries and lists are traversed recursively.
        
    Returns:
        Any: 
            A version of `x`, but with tensors
            and tensors contained in nested dictionaries/lists 
            recursively moved to CPU.
    '''
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, dict):
        return {key: recursive_to_cpu(value) for key, value in x.items()}
    elif isinstance(x, list):
        return [recursive_to_cpu(value) for value in x]
    else:
        return x
    

def apply_agg(x: torch.Tensor, agg: Aggregation) -> torch.Tensor:
    '''
    Applies an aggregation function `(mean, median, min, max)` to a tensor.

    Args:
        x (torch.Tensor): 
            The tensor to aggregate.
        agg (Aggregation): 
            The aggregation function to apply.
            Supports: `mean`, `median`, `min`, `max`.

    Returns:
        torch.Tensor: 
            The aggregated value from applying `agg` to `x`.
    '''
    x = x.float() # Ensure float

    if agg == 'mean':
        return x.mean()
    elif agg == 'median':
        return x.median()
    elif agg == 'min':
        return x.min()
    elif agg == 'max':
        return x.max()
    else:
        raise ValueError(f'Unknown aggregation: {agg}')