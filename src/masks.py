#####################################
# Imports & Dependencies
#####################################
import torch

from typing import Dict
from src.ml_types import RGBTuple


#####################################
# Functions
#####################################
def rgb_to_key(rgb: RGBTuple) -> int:
    '''
    Converts a RGB-tuple to a base-10 integer key using the formula
    n = (256**2) * r + (256 * g) + b

    Note: Values of the RGB tuple should be in [0, 255].
    '''
    is_rgb_tuple = (        
        isinstance(rgb, tuple)
        and len(rgb) == 3
        and all(0 <= x <= 255 for x in rgb)
        and all(isinstance(x, int) for x in rgb)
    )
    if not is_rgb_tuple:
        raise ValueError('Input must be a RGB tuple.')
        
    return (256**2 * rgb[0]) + (256 * rgb[1]) + rgb[2]
    

def key_to_rgb(n: int) -> RGBTuple:
    '''
    Converts a base-10 integer key into a RGB tuple, by interpreting it as 
    n = (256**2) * r + (256 * g) + b

    Note: Values of the RGB tuple are in [0, 255].
    '''
    if not (0 <= n < 256**3):
        raise ValueError(f'Input must be an integer in [0, 256^3 - 1].')
        
    r = n // 256**2
    g = (n // 256) % 256
    b = n % 256

    return (r, g, b)


def rgb_to_idx_mask(
    rgb_mask: torch.Tensor, 
    rgb_to_idx: Dict[RGBTuple, int], 
    fill_idx: int = -100
) -> torch.Tensor:
    '''
    Converts a RGB segmentation mask into an index segmentation mask.

    Note: The values of all RGB tuples should be in [0, 255].

    Args:
        rgb_mask (torch.Tensor): RGB segmentation mask.
                                 This is a 3D tensor of shape (3, height, width),
                                 where each pixel is an RGB tuple with values in [0, 255].
        rgb_to_idx (Dict[RGBTuple, int]): Dictionary mapping RGB tuples to integer indices.
                                          This mapping should be one-to-one (injective).
        fill_idx (int): Integer index used to fill in pixels whose RGB tuple 
                        is not present in `rgb_to_idx`.
                        Default is `-100`.
    Returns:
        torch.Tensor: Index segmentation mask.
                      This is a 2D tensor of shape (height, width), 
                      where each pixel is an integer.
    '''
    if (rgb_mask.ndim != 3) or (rgb_mask.shape[0] != 3):
        raise ValueError('Expected rgb_mask to be a 3D tensor of shape (3, height, width).')

    rgb_pixels = rgb_mask.view(3, -1) # Shape: (3, height * width)

    # Treat RGB-tuples as base-256 and convert to base-10
        # This gives an integer key for each RGB-tuple
    rgb_keys = (256**2 * rgb_pixels[0]) + (256 * rgb_pixels[1]) + rgb_pixels[2] # Shape: (height * width,)

    idx_pixels = torch.full_like(rgb_keys, fill_value = fill_idx, dtype = torch.long)

    # Fill in index where key is matched by rgb_keys
    for rgb, idx in rgb_to_idx.items():
        key = rgb_to_key(rgb) # Convert to integer key
        idx_pixels[rgb_keys == key] = idx
    
    h, w = rgb_mask.shape[-2:]
    return idx_pixels.reshape(h, w)


def idx_to_rgb_mask(
    idx_mask: torch.Tensor, 
    idx_to_rgb: Dict[int, RGBTuple], 
    fill_rgb: RGBTuple = (114, 114, 114)
) -> torch.Tensor:
    '''
    Converts index segmentation mask into a RGB segmentation mask.

    Note: The values of all RGB tuples should be in [0, 255].

    Args:
        idx_mask (torch.Tensor): Index segmentation mask.
                                 This is a 2D tensor of shape (height, width), 
                                 where each pixel is an integer.
        idx_to_rgb (Dict[int, RGBTuple]): Dictionary mapping integer indices to RGB tuples.
                                          This mapping should be one-to-one (injective).
        fill_rgb (RGBTuple): RGB tuple used to fill in pixels whose index 
                             is not present in `idx_to_rgb`.
                             Default is (114, 114, 114).
    Returns:
        torch.Tensor: RGB segmentation mask.
                      This is a 3D tensor of shape (3, height, width),
                      where each pixel is an RGB tuple with values in [0, 255].
    '''
    if idx_mask.ndim != 2:
        raise ValueError('Expected idx_mask to be a 2D tensor of shape (height, width).')

    # Treat RGB-tuples as base-256 and convert to base-10
        # This gives an integer key for each RGB-tuple
    fill_key = rgb_to_key(fill_rgb)
    rgb_keys = torch.full_like(idx_mask, fill_value = fill_key, dtype = torch.long) # Shape: (height, width)

    # Fill in where index matches, with the integer key
    for idx, rgb in idx_to_rgb.items():
        rgb_keys[idx_mask == idx] = rgb_to_key(rgb)

    # Separate integer key into RGB values (converting base-10 to base-256)
    r_values = rgb_keys // 256**2
    g_values = (rgb_keys // 256) % 256
    b_values = rgb_keys % 256

    return torch.stack([r_values, g_values, b_values]).to(torch.uint8)