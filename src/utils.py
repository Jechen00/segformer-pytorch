#####################################
# Imports & Dependencies
#####################################
import torch

import os
import warnings
import random
import numpy as np 

from typing import Union, Any, Literal


#####################################
# Functions
#####################################
def make_tuple(x: Union[Any, tuple]) -> tuple:
    '''
    Converts input to a tuple `(x, x)`, if it is not already a tuple. 

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

def nested_extract(nested_dict: dict, path: str, strict: bool = True, default: Any = None) -> Any:
    '''
    Extracts a value from a nested dictionary given a dot-separated path.
    Example:
        The path `key1.key2.key3` returns `nested_dict['key1']['key2']['key3']`.

    Args:
        nested_dict (dict): Nested dictionary to extract from.
        path (str): Dot-separated key path consisting of only keys in `nested_dict`.
        strict (bool): If `True`, raises a `KeyError` on missing keys or when an intermediate value is not a dictionary.
                       If `False`, does not raise any errors and instead returns a `default` value.
        default (Any): A default value to return when encountering missing keys and `strict=False`.
                       Default is None.
    Returns:
        Any: The extracted value from nested_dict after traversing through path.
    '''
    value = nested_dict
    for key in path.split('.'):
        if (not isinstance(value, dict)):
            if strict:
                raise KeyError(
                    f"Expected dictionary, but got {type(value)} at key '{key}' in path '{path}'."
                )
            return default
        elif key not in value:
            if strict:
                raise KeyError(f"Missing key '{key}' in path '{path}'.")
            return default
            
        value = value[key]
    return value
    
def apply_agg(
        x: Union[np.ndarray, torch.Tensor], 
        agg: Literal['mean', 'min', 'max']
) -> float:
    '''
    Applies an aggregation function `(mean, min, max)` to a numpy array or tensor.

    Args:
        x (np.ndarray): The numpy array or tensor to aggregate.
        agg (Literal['mean', 'min', 'max']): The aggregation function to apply `(mean, min, max)`.

    Returns:
        float: The aggregated value from applying `agg` to `x`.
    '''
    if isinstance(x, torch.Tensor):
        x = x.float()
    elif isinstance(x, np.ndarray):
        x = x.astype(np.float32)

    if agg == 'mean':
        return x.mean().item()
    elif agg == 'min':
        return x.min().item()
    elif agg == 'max':
        return x.max().item()
    else:
        raise ValueError(f'Unknown aggregation: {agg}')
    
def recursive_to_cpu(x: Any) -> Any:
    '''
    Recursively moves all tensors in a nested dictionary/list structure to the CPU.
    Other objects (e.g. floats, ints, numpy arrays) remain unchanged.
    
    Args:
        x (Any): Any Python object. Tensors are moved to CPU.
                 Dictionaries and lists are traversed recursively.
        
    Returns:
        Any: The same object as the input `x`, but with all tensors moved to CPU.
    '''
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, dict):
        return {key: recursive_to_cpu(value) for key, value in x.items()}
    elif isinstance(x, list):
        return [recursive_to_cpu(value) for value in x]
    else:
        return x