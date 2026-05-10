#####################################
# Imports & Dependencies
#####################################
import torch

import os
from pathlib import Path
import random
import numpy as np 
from numbers import Real
from PIL import Image

from typing import (
    Union, Any, Literal, Optional, List, Dict, Tuple
)

from src.ml_types import ImageInput, Sample, BatchedSamples


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
        seed (int): The seed value to set.
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


def make_tuple(x: Union[Any, tuple], num_rep: int = 2) -> tuple:
    '''
    Converts a non-tuple input into a repeated tuple.
    If input is a tuple, it is unchanged.

    Args:
        x (Union[Any, tuple]): Input value. If not a tuple, it will be repeated.
        num_rep (int): Number of repetitions for the output tuple,
                       when `x` is a not a tuple.

    Returns:
        tuple: If `x` is not a tuple, returns `(x, x, ..., x)` of length `num_rep`
               If `x` is a tuple, returns `x` unchanged.
    '''
    if not isinstance(x, tuple):
        return (x,) * num_rep
    else:
        return x


def make_range(x: Any) -> Union[Tuple[Real, Real], Any]:
    '''
    Converts a numeric value `x` into `(-x, x)`, representing a range of values.
    If the input is not an integer or float, then the input is returned unchanged.
    '''
    if isinstance(x, Real):
        return (-x, x)
    else:
        return x

      
def all_or_none(*params) -> bool:
    '''
    Checks if params are all None or all provided (not None).
    '''
    return all(p is None for p in params) or all(p is not None for p in params)


def check_tensor_shapes(
    tensors: List[torch.Tensor], 
    context_name: str = 'inputs'
) -> None:
    '''
    Checks that a list of inputs are all tensors and that they all have the same shape.

    Args:
        tensors (List[torch.Tensor]): A list of tensors of the same shape.
        context_name (str): Name for the elements in `tensors` to provide context on what they are.
                            This is used for more specific error messages.
                            Default is `inputs`.
    '''
    ref_shape = None
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f'Expected all {context_name} to be tensors. Got: {type(tensor)}'
            )

        tensor_shape = tensor.shape
        if ref_shape is None:
            ref_shape = tensor_shape
        elif tensor_shape != ref_shape:
            raise ValueError(
                f'Expected all {context_name} to have the same shape. '
                f'Got: {tuple(ref_shape)} and {tuple(tensor_shape)}'
            )

 
def get_img_size(img: ImageInput) -> Tuple[int, int]:
    '''
    Gets the spatial size (height, width) of and image.

    Args:
        img (ImageInput): A PIL image or tensor.
                          If `torch.Tensor`, shape should be (..., height, width).

    Returns:
        Tuple[int, int]: Tuple representing (height, width) of `img`.
    '''
    if isinstance(img, torch.Tensor):
        return img.shape[-2:] # height, width
    elif isinstance(img, Image.Image):
        return img.size[::-1] # PIL gives (width, height) --> need to flip
    else:
        raise TypeError('Expected PIL image or tensor.')

   
def extract_imgs(samps: Union[Sample, BatchedSamples]) -> Union[ImageInput, List[ImageInput]]:
    '''
    Gets all images from a sample or batch of samples.

    Args:
        samps (Union[Sample, BatchedSamples]): Sample or batch of samples containing image information.
            Supports:
                - A single image (PIL image or tensor)
                - A single-sample dictionary, where the 'image' key contains a single image
                - A list of images (PIL image or tensor)
                - A batched-sample dictionary, where the 'image' key contains a list of images
                - A collated tensor, e.g. of shape (batch_size, channels, height, width)

    Returns:
        Union[ImageInput, List[ImageInput]: The extracted images from `samps`.
                                            The structure depends on the type of input.
                                            It will either be a single PIL image or tensor (possibly batched)
                                            or a list of PIL images or tensors.
    '''
    if isinstance(samps, dict):
        imgs = samps['image']
    elif isinstance(samps, list):
        imgs = [s['image'] if isinstance(s, dict) else s for s in samps]
    else:
        imgs = samps  
    return imgs


def normalize_file_path(file_path: Union[str, Path], path_name: Optional[str] = None) -> Path:
    '''
    Normalize and validate a file path.
    This converts all `path` inputs into `pathlib.Path` objects.
    It also checks that `path` contains a file extension 
    and does not end with a path separator ('/' or '\\').

    Args:
        file_path (Union[str, Path]): The path to normalize and validate.
        path_name (optional, str): Name of `file_path` to use for error messages.

    Returns:
        Path: The validated `pathlib.Path` object.
    '''
    path_name = 'path' if path_name is None else path_name

    path = Path(file_path)
    if str(path).endswith(('/', '\\')):
        raise ValueError(
            f"{path_name} must not end with a path separator ('/' or '\\'). Got: {file_path}"
        )
    
    if path.suffix == '':
        raise ValueError(
            f'{path_name} must end with a file extension. Got: {file_path}'
        )
    
    return path


def nested_extract(nested_dict: dict, key_path: str, strict: bool = True, default: Any = None) -> Any:
    '''
    Extracts a value from a nested dictionary given a dot-separated key path.
    Example:
        The key path `key1.key2.key3` returns `nested_dict['key1']['key2']['key3']`.

    Args:
        nested_dict (dict): Nested dictionary to extract from.
        key_path (str): Dot-separated key path consisting of only keys in `nested_dict`.
        strict (bool): If `True`, raises a `KeyError` on missing keys or when an intermediate value is not a dictionary.
                       If `False`, does not raise any errors and instead returns a `default` value.
        default (Any): A default value to return when encountering missing keys and `strict=False`.
                       Default is None.
    Returns:
        Any: The extracted value from nested_dict after traversing through `key_path`.
    '''
    value = nested_dict
    for key in key_path.split('.'):
        if (not isinstance(value, dict)):
            if strict:
                raise KeyError(
                    f"Expected dictionary, but got {type(value)} at key '{key}' in key_path '{key_path}'."
                )
            return default
        elif key not in value:
            if strict:
                raise KeyError(f"Missing key '{key}' in key_path '{key_path}'.")
            return default
            
        value = value[key]
    return value


def transpose_list_dict(
    data: Union[List[Dict[str, Any]], Dict[str, List[Any]]], 
    mode: Literal['to_cols', 'to_rows'] = 'to_cols'
) -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    '''
    Transposes a list of dictionaries into a dictionary of lists, and vice versa.
    
    Args:
        data (Union[List[Dict[str, Any]], Dict[str, List[Any]]]):
            A list of dictionaries or a dictionary of lists, depending on `mode`.
                - `mode = 'to_cols'`: List of dictionaries, 
                                      All dictionaries are expected to have the same keys.
                - `mode = 'to_rows'`: Dictionary of lists.
                                      All lists are expected to have the same length.

        mode (Literal['to_cols', 'to_rows']): The mode of transpose.
            - `to_cols`: Transposes a list of dictionaries into a dictionary of lists.
            - `to_rows`: Transposes a dictionary of lists into a list of dictionaries.
        
    Returns:
        Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
            - `mode = 'to_cols'`: Dictionary of lists, all with the same length.
            - `mode = 'to_rows'`: List of dictionaries, all with the same keys.
    '''
    if mode == 'to_cols':
        if not isinstance(data, list):
            raise TypeError(
                f"data must be a list of dictionaries if mode = 'to_cols'. Got: {type(data)}"
            )
            
        return {
            key: [data_dict[key] for data_dict in data]
            for key in data[0].keys()
        }
            
    elif mode == 'to_rows':
        if not isinstance(data, dict):
            raise TypeError(
                f"data must be a dictionary of lists if mode = 'to_rows'. Got: {type(data)}"
            )
            
        keys = list(data.keys())
        values = list(data.values())
        return [
            {k: v[i] for (k, v) in zip(keys, values)}
            for i in range(len(values[0]))
        ]
            
    else:
        raise ValueError(
            f"mode must be 'to_cols' or 'to_rows'. Got: {mode}"
        )


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