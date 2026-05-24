#####################################
# Imports & Dependencies
#####################################
import torch
from typing import List, Tuple, Optional


#####################################
# Helper Functions
#####################################
def ensure_batched(
    tensor: torch.Tensor,
    unbatched_ndim: int,
    context_name: str = 'inputs'
) -> torch.Tensor:
    '''
    Ensures a tensor has a batch dimension.
    If the input tensor is unbatched, a batch dimension is prepended.
    If it is already batched, it is returned unchanged.

    Args:
        tensor (torch.Tensor): The tensor to check and format to batched.
        unbatched_ndim (int): The number of dimensions for an unbatched tensor.
                              A batched tensor is assumed to have `(unbatched_ndim+1)` dimensions.
        context_name (str): Name for `tensor` to provide context on what it represents.
                            This is for more specific error messages.
                            Default is `inputs`.
    '''
    batched_ndim = unbatched_ndim + 1

    ndim = tensor.ndim
    if ndim == unbatched_ndim:
        return tensor.unsqueeze(0) # Add a batched dimension
    elif ndim == batched_ndim:
        return tensor
    else:
        raise ValueError(
            f'Expected {context_name} to be '
            f'{unbatched_ndim}D or {batched_ndim}D tensors. '
            f'Got a {ndim}D tensor.'
        )   


#####################################
# Validation Functions
#####################################
def _validate_ndim(tensor: torch.Tensor, ndim: int, context_name: str = 'inputs') -> None:
    '''
    Validates a tensor has a specified number of dimensions (ndim).

    Args:
        tensor (torch.Tensor): The tensor to validate.
        ndim (int): The expected number of dimensions for `tensor`.
        context_name (str): Name for `tensor` to provide context on what it represents.
                            This is for more specific error messages.
                            Default is `inputs`.
    '''
    if tensor.ndim != ndim:
        raise ValueError(
            f'Expected {context_name} to be {ndim}D tensors. '
            f'Got a {tensor.ndim}D tensor.'
        )


def _validate_channel_size(
    tensor: torch.Tensor,
    channel_size: int,
    context_name: str = 'inputs'
) -> None:
    '''
    Validates the channel dimension size of a tensor.
    Assumes the tensor is in CHW format.

    Args:
        tensor (torch.Tensor): The tensor to validate.
        channel_size (int): The expected channel dimension size for `tensor`
        context_name (str): Name for `tensor` to provide context on what it represents.
                            This is for more specific error messages.
                            Default is `inputs`.
    '''
    if tensor.shape[-3] != channel_size:
        raise ValueError(
            f'Expected {context_name} to be tensors '
            f'with channel dimension of size {channel_size} and in CHW format. '
            f'Got a tensor of shape {tuple(tensor.shape)}'
        )
    

def _validate_shape(tensor: torch.Tensor, shape: Tuple[int, ...], context_name: str = 'inputs') -> None:
    '''
    Validates a tensor has a specified shape.

    Args:
        tensor (torch.Tensor): The tensor to validate.
        shape (Tuple[int, ...]): The expected shape for `tensor`.
        context_name (str): Name for `tensor` to provide context on what it represents.
                            This is for more specific error messages.
                            Default is `inputs`.
    '''
    if tensor.shape != shape:
        raise ValueError(
            f'Expected {context_name} to be tensors of shape {tuple(shape)}. '
            f'Got a tensor with shape {tuple(tensor.shape)}.'
        )


def _validate_same_shape(
    tensors: List[torch.Tensor], 
    shape: Optional[Tuple[int, ...]] = None,
    context_name: str = 'inputs'
) -> None:
    '''
    Validates that a list of inputs are all tensors and that they are all the same shape.
    Optionally checks that they have a specified shape.

    Args:
        tensors (List[torch.Tensor]): A list of tensors to validate.
        shape (optional, Tuple[int, ...]): Expected shape of each tensor.
                                           If not provided, the expected shape will be set 
                                           to the first tensor in `tensors`.
        context_name (str): Name for the elements in `tensors` to provide context on what they represents.
                            This is for more specific error messages.
                            Default is `inputs`.
    '''
    if shape is not None:
        err_prefix = f'Expected {context_name} to be tensors of shape {tuple(shape)}. '
    else:
        err_prefix = f'Expected {context_name} to be tensors of the same shape. '

    # Check tensor shapes
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f'Expected {context_name} to be tensors. Got: {type(tensor)}'
            )

        tensor_shape = tensor.shape
        if shape is None:
            shape = tensor_shape
        if tensor.shape != shape:
            raise ValueError(
                f'{err_prefix} '
                f'Got tensors of shape {tuple(shape)} and {tuple(tensor.shape)}.'
            )