#####################################
# Imports & Dependencies
#####################################
import torch
import numpy as np 
import random
import os

from numbers import Real
from typing import Union, Any, Literal


#####################################
# Functions
#####################################
def make_tuple(x: Union[Any, tuple]) -> tuple:
    '''
    Converts input to a tuple (x, x), if it is not already a tuple. 

    Args:
        x (Union[Any, tuple]): Input to convert into a tuple (if needed).
    '''
    if not isinstance(x, tuple):
        return (x, x)
    else:
        return x
    
def all_or_none(*params) -> bool:
    '''
    Checks if params are all None or all provided (not None).
    '''
    return all(p is None for p in params) or all(p is not None for p in params)

def set_seed(seed: int = 0):
    '''
    Sets random seed and deterministic settings 
    for reproducibility across:
        - PyTorch
        - NumPy
        - Python's random module
        - CUDA
    
    Args:
        seed (int): The seed value to set.
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

def apply_agg(arr: np.ndarray, agg: Literal['mean', 'min', 'max']) -> Real:
    '''
    Applies an aggregation function (mean, min, max) to a numpy array.

    Args:
        arr (np.ndarray): The numpy array to aggregate.
        agg (Literal['mean', 'min', 'max']): The aggregation function to apply (mean, min, max).

    Returns:
        Real: A real numeric value from aggregating the numpy array.
    '''
    if agg == 'mean':
        return arr.mean()
    elif agg == 'min':
        return arr.min()
    elif agg == 'max':
        return arr.max()
    else:
        raise ValueError(f'Unknown aggregation: {agg}')