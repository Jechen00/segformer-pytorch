#####################################
# Imports & Dependencies
#####################################
import torch
import numpy as np 
import random
import os

from typing import Union, Any


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