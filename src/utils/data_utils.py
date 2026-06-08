#####################################
# Imports & Dependencies
#####################################
import torch

import numpy as np 
from PIL import Image

from typing import Union, Any, Literal, List, Dict, Tuple

from src.ml_types import PythonNum, ImageInput, IndexLike


#####################################
# Functions
#####################################
def make_tuple(x: Union[Any, tuple], num_rep: int = 2) -> tuple:
    '''
    Converts a non-tuple input into a repeated tuple.
    If input is a tuple, it is unchanged.

    Args:
        x (Union[Any, tuple]): 
            Input value. If not a tuple, it will be repeated.
        num_rep (int): 
            Number of repetitions for the output tuple, when `x` is a not a tuple.

    Returns:
        tuple: 
            If `x` is not a tuple, returns `(x, x, ..., x)` of length `num_rep`
            If `x` is a tuple, returns `x` unchanged.
    '''
    if not isinstance(x, tuple):
        return (x,) * num_rep
    else:
        return x


def make_range(x: Any) -> Union[Tuple[PythonNum, PythonNum], Any]:
    '''
    Converts a numeric value `x` into `(-x, x)`, representing a range of values.
    If the input is not an integer or float, then the input is returned unchanged.
    '''
    if isinstance(x, PythonNum):
        return (-x, x)
    else:
        return x

      
def all_or_none(*params) -> bool:
    '''
    Checks if `params` are all `None` or all not `None` (all provided).
    '''
    return all(p is None for p in params) or all(p is not None for p in params)

 
def get_img_size(img: ImageInput) -> Tuple[int, int]:
    '''
    Gets the spatial size `(height, width)` of and image.

    Args:
        img (ImageInput): 
            A PIL image or tensor.
            If tensor, shape should be `(..., height, width)`.

    Returns:
        Tuple[int, int]: 
            Tuple representing `(height, width)` of `img`.
    '''
    if isinstance(img, torch.Tensor):
        h, w = img.shape[-2:] # height, width
    elif isinstance(img, Image.Image):
        w, h = img.size # PIL gives (width, height)
    else:
        raise TypeError(f'Expected PIL image or tensor. Got: {type(img)}')
    
    return (h, w)


def format_idxs(idxs: IndexLike) -> Union[int, List[int]]:
    '''
    Formats an `IndexLike` so that it only contains Python integers.
    Specifically, it converts to a single integer or a list of integers.
    
    Args:
        idxs (IndexLike): 
            Indices to format.
            This must be one of:
                - A single integer
                - A list of integers
                - A ndarray of integers (single-element or 1D)
                - A tensor of integers (single-element or 1D)

    Returns:
        Union[int, List[int]]: 
            Formatted indices.
            This is an integer if `idxs` was an integer or a single-element ndarray/tensor.
            Otherwise, this is a list of integers.
    '''
    # Integer input
    if type(idxs) is int:
        return idxs 
    
    # Numpy input
    elif isinstance(idxs, np.ndarray):
        if idxs.size == 1:
            return idxs.item()
        elif idxs.ndim == 1:
            idxs = idxs.tolist()  # Need to check all elements are integers
        else:
            raise ValueError('NumPy ndarray indices must be single-element (size = 1) or 1D.')
        
    # Tensor input
    elif isinstance(idxs, torch.Tensor):
        if idxs.numel() == 1:
            return idxs.item()
        elif idxs.ndim == 1:
            idxs = idxs.tolist() # Need to check all elements are integers
        else:
            raise ValueError('Tensor indices must be single-element (numel() = 1) or 1D.')

    elif not isinstance(idxs, list):
        raise TypeError(
            'Expected idxs to be an integer, list, ndarray, or tensor. '
            f'Got: {type(idxs)}'
        )
    
    # Check all elements in a list, ndarray, or tensor are integers
    if not all((type(idx) is int) for idx in idxs):
        raise TypeError(
            'If idxs is a list, ndarray, or tensor, '
            'all elements must be integers.'
        )
    return idxs


def transpose_list_dict(
    data: Union[List[Dict[str, Any]], Dict[str, List[Any]]], 
    mode: Literal['to_cols', 'to_rows'] = 'to_cols'
) -> Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
    '''
    Transposes a list of dictionaries into a dictionary of lists, and vice versa.
    
    Args:
        data (Union[List[Dict[str, Any]], Dict[str, List[Any]]]):
            A list of dictionaries or a dictionary of lists, depending on `mode`.
                - `mode='to_cols'`: List of dictionaries, 
                                    All dictionaries are expected to have the same keys.
                - `mode='to_rows'`: Dictionary of lists.
                                    All lists are expected to have the same length.

        mode (Literal['to_cols', 'to_rows']): 
            The mode of transpose.
                - `to_cols`: Transposes a list of dictionaries into a dictionary of lists.
                - `to_rows`: Transposes a dictionary of lists into a list of dictionaries.
        
    Returns:
        Union[List[Dict[str, Any]], Dict[str, List[Any]]]:
            - `mode='to_cols'`: Dictionary of lists, all with the same length.
            - `mode='to_rows'`: List of dictionaries, all with the same keys.
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


def inverse_mapping(mapping: dict) -> dict:
    '''
    Returns the inverse (value-key) of a key-value mapping dictionary.
    The mapping must be injective (one-to-one).
    '''
    inverse = {}
    for key, value in mapping.items():
        if value not in inverse:
            inverse[value] = key
        else:
            raise ValueError(
                f'Duplicate value found in mapping: {value}. '
                'Please ensure that the mapping an injective (one-to-one).'
            )
            
    return inverse


def nested_extract(nested_dict: dict, key_path: str, strict: bool = True, default: Any = None) -> Any:
    '''
    Extracts a value from a nested dictionary given a dot-separated key path.
    Example:
        The key path `key1.key2.key3` returns `nested_dict['key1']['key2']['key3']`.

    Args:
        nested_dict (dict): 
            Nested dictionary to extract from.
        key_path (str): 
            Dot-separated key path consisting of only keys in `nested_dict`.
        strict (bool): 
            If `True`, raises a `KeyError` on missing keys or when an intermediate value is not a dictionary.
            If `False`, does not raise any errors and instead returns a `default` value.
        default (Any): 
            A default value to return when encountering missing keys and `strict=False`.
            Default is `None`.
    Returns:
        Any: 
            The extracted value from `nested_dict` after traversing through `key_path`.
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


