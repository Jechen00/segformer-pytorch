#####################################
# Imports & Dependencies
#####################################
import torch

from typing import Dict, List
from src.ml_types import RGBTuple


#####################################
# RGB Tuple Conversion Functions
#####################################
def rgb_to_key(rgb: RGBTuple) -> int:
    '''
    Converts a RGB-tuple to a base-10 integer key using the formula
    n = (256**2) * r + (256 * g) + b

    Note: Values of the RGB tuple should be in [0, 255].
    '''
    if not is_rgb_tuple(rgb):
        raise ValueError('Input must be a tuple of 3 integers in [0, 255].')
    
    r, g, b = rgb
    return (256**2 * r) + (256 * g) + b
    

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


#####################################
# Tensor Mask Conversion Functions
#####################################
def rgb_to_idx_mask(
    rgb_masks: torch.Tensor, 
    rgb_to_idx: Dict[RGBTuple, int], 
    fill_idx: int = -100
) -> torch.Tensor:
    '''
    Converts RGB segmentation masks into index segmentation masks.

    Note: The values of all RGB tuples should be in [0, 255].

    Args:
        rgb_masks (torch.Tensor): RGB segmentation masks.
                                  This is either a 3D tensor `(3, height, width)`,
                                  or a 4D tensor `(batch_size, 3, height, width)`.
                                  Each pixel is a RGB tuple with values in [0, 255].
        rgb_to_idx (Dict[RGBTuple, int]): Dictionary mapping RGB tuples to integer indices.
                                          This mapping should be one-to-one (injective).
        fill_idx (int): Integer index used to fill in pixels whose RGB tuple 
                        is not present in `rgb_to_idx`.
                        Default is `-100`.
    Returns:
        torch.Tensor: Index segmentation masks.
                      If `rgb_masks` is a 3D tensor, this is a 2D tensor of shape `(height, width)`.
                      If `rgb_masks` is a 4D tensor, this is a 3D tensor of shape `(batch_size, height, width)`.
                      Each pixel is an integer index.
    '''
    # Separate R, G, B channels
    r, g, b = torch.unbind(rgb_masks.long(), dim = -3)

    # Treat RGB-tuples as base-256 and convert to base-10
        # This gives an integer key for each RGB-tuple
     # Shape: (height, width) or (batch_size, height, width)
    rgb_keys = (256**2 * r) + (256 * g) + b

    # Shape: (height, width) or (batch_size, height, width)
    idx_masks = torch.full_like(rgb_keys, fill_value = fill_idx, dtype = torch.long)

    # Fill in index where key is matched by rgb_keys
    for rgb, idx in rgb_to_idx.items():
        key = rgb_to_key(rgb) # Convert to integer key
        idx_masks[rgb_keys == key] = idx
    
    return idx_masks


def idx_to_rgb_mask(
    idx_masks: torch.Tensor, 
    idx_to_rgb: Dict[int, RGBTuple], 
    fill_rgb: RGBTuple = (114, 114, 114)
) -> torch.Tensor:
    '''
    Converts index segmentation masks into a RGB segmentation masks.

    Note: The values of all RGB tuples should be in [0, 255].

    Args:
        idx_masks (torch.Tensor): Index segmentation masks.
                                  This is either a 2D tensor `(height, width)`
                                  or a 3D tensor `(batch_size, height, width)`.
                                  Each pixel should be an integer.
        idx_to_rgb (Dict[int, RGBTuple]): Dictionary mapping integer indices to RGB tuples.
                                          This mapping should be one-to-one (injective).
        fill_rgb (RGBTuple): RGB tuple used to fill in pixels whose index 
                             is not present in `idx_to_rgb`.
                             Default is `(114, 114, 114)`.
    Returns:
        torch.Tensor: RGB segmentation masks.
                      If `idx_masks` was a 2D tensor, this is a 3D tensor of shape `(3, height, width)`.
                      If `idx_masks` was a 3D tensor, this is a 4D tensor of shape `(batch_size, 3, height, width)`.
                      Each pixel is a RGB tuple with values in [0, 255].
    '''
    # Treat RGB-tuples as base-256 and convert to base-10
        # This gives an integer key for each RGB-tuple
    # Shape: (height, width) or (batch_size, height, width)
    fill_key = rgb_to_key(fill_rgb)
    rgb_keys = torch.full_like(idx_masks, fill_value = fill_key, dtype = torch.long)

    # Fill in where index matches, with the integer key
    for idx, rgb in idx_to_rgb.items():
        rgb_keys[idx_masks == idx] = rgb_to_key(rgb)

    # Separate integer key into RGB values (converting base-10 to base-256)
    r = rgb_keys // 256**2
    g = (rgb_keys // 256) % 256
    b = rgb_keys % 256

    # Shape: (3, height, width) or (batch_size, 3, height, width)
    rgb_mask = torch.stack([r, g, b], dim = -3).to(torch.uint8)

    return rgb_mask


def rgb_to_visibility_mask(
    rgb_masks: torch.Tensor,
    visible_rgbs: List[RGBTuple]
) -> torch.Tensor:
    '''
    Adds an alpha channel to RGB segmentation masks (convert it to RGBA).
    This alpha channel is set to 255 (fully visible) for specified RGB colors,
    and 0 (invisible) elsewhere.

    Args:
        rgb_masks (torch.Tensor): RGB segmentation masks.
                                  This is either a 3D tensor `(3, height, width)`,
                                  or a 4D tensor `(batch_size, 3, height, width)`.
                                  Each pixel is a RGB tuple with values in [0, 255].

        visible_rgbs (List[RGBTuple]): List of RGB tuples to set as visible in the alpha channel.
                                       The values of each tuple must be in [0, 255].
                                    
    Returns:
        torch.Tensor: RGBA segmentation masks.
                      If `rgb_masks` is a 3D tensor, this returns a 3D tensor `(4, height, width)`.
                      If `rgb_masks` is a 4D tensor, this returns a 4D tensor `(batch_size, 4, height, width)`.
    '''
    # Separate R, G, B channels
    r, g, b = torch.unbind(rgb_masks.long(), dim = -3)

    # Treat RGB-tuples as base-256 and convert to base-10
        # This gives an integer key for each RGB-tuple
     # Shape: (height, width) or (batch_size, height, width)
    rgb_keys = (256**2 * r) + (256 * g) + b

    # Initialize alpha mask to invisible
        # This is initially treated as a boolean mask over spatial dims
    alpha_masks = torch.zeros_like(rgb_keys, dtype = torch.bool)

    # Determine which pixels need to visible
    for rgb in visible_rgbs:
        key = rgb_to_key(rgb) # Convert to integer key
        alpha_masks = alpha_masks | (rgb_keys == key)

    # Convert boolean mask to a single-channel alpha tensor
    # Shape: (1, height, width) or (batch_size, 1, height, width)
    alpha_masks = (alpha_masks * 255).to(dtype = torch.uint8).unsqueeze(-3)
    
    return torch.concat([rgb_masks, alpha_masks], dim = -3)


#####################################
# Validation Functions
#####################################
def is_rgb_tuple(rgb: RGBTuple) -> bool:
    '''
    Returns whether `rgb` is a RGB tuple.
    This only returns `True` when:
        1. `rgb` is a tuple
        2. `rgb` has a length of 3
        3. All values in `rgb` are in the range [0, 255]
        4. All values in `rgb` are integers
    '''
    return (        
        isinstance(rgb, tuple)
        and len(rgb) == 3
        and all(0 <= x <= 255 for x in rgb)
        and all(type(x) is int for x in rgb)
    )