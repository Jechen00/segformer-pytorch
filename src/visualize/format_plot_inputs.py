#####################################
# Imports & Dependencies
#####################################
import torch
from torchvision.transforms.v2 import functional as F

import numpy as np
from PIL import Image

from typing import Dict, List, Literal, Optional, Union, Tuple, TypeAlias

from src.masks import idx_to_rgb_mask, rgb_to_visibility_mask
from src.utils.data_utils import all_or_none
from src.utils.shape_utils import ensure_batched, _validate_channel_size, _validate_ndim
from src.ml_types import ImageInput, ImageLabel, RGBTuple

NPImageList: TypeAlias = List[np.ndarray]
IntLabelList: TypeAlias = List[int]


#####################################
# Specific Task Format Functions
#####################################
def format_cls(
    imgs: Union[ImageInput, List[ImageInput]],
    pred_labels: torch.Tensor,
    targ_labels: Optional[Union[ImageLabel, List[ImageLabel]]] = None,
) -> Tuple[NPImageList, IntLabelList, Optional[IntLabelList]]:
    '''
    Formats images, prediction labels, and optional target labels 
    for image classification plotting.

    The formatted outputs are used by the image classification figure function:
        - `visualize.figures.make_cls_figure`

    Note: Labels must be integer class indices (not one-hot).

    Args:
        imgs (Union[ImageInput, List[ImageInput]]):
            Images to format.
            This supports:
                - A single PIL image
                - A single 3D tensor of shape (channels, height, width)
                - A list of images (each element is a PIL image or 3D tensor)
                - A batched 4D tensor of shape (batch_size, channels, height, width)
        pred_labels (torch.Tensor): 
            Prediction labels to format.
            This is a 1D tensor of shape (batch_size,).
        targ_labels (optional, Union[ImageLabel, List[ImageLabel]]):
            Target labels to format.
            This supports:
                - A single integer
                - A single-element tensor
                - A list of labels (each element is an integer or single-element tensor)
                - A batched 1D tensor of shape (batch_size,)

    Returns:
        imgs (NPImageList): 
            Images formatted to a list of ndarrays in HWC format.
        pred_labels (IntLabelList): 
            Prediction labels formatted to a list of integers.
        targ_labels (optional, IntLabelList): 
            Target labels formatted to a list of integers.
            This is `None` if `targ_labels` was not provided as input.
    '''
    # Format images
    imgs = format_images(imgs)

    # Format prediction labels
    try:
        pred_labels = format_labels(pred_labels)
    except Exception as e:
        raise ValueError('Unable to format prediction labels (pred_labels)') from e

    # Format target labels if provided
    if targ_labels is not None:
        try:
            targ_labels = format_labels(targ_labels)
        except Exception as e:
            raise ValueError('Unable to format target labels (targ_labels)') from e
    
    return imgs, pred_labels, targ_labels


def format_seg(
    imgs: Union[ImageInput, List[ImageInput]],
    pred_masks: torch.Tensor,
    idx_to_rgb: Dict[int, RGBTuple],
    targ_masks: Optional[Union[ImageInput, List[ImageInput]]] = None,
    targ_mode: Optional[Literal['index', 'rgb']] = None,
    fill_rgb: RGBTuple = (114, 114, 114),
    visible_rgbs: Optional[List[RGBTuple]] = None
) -> Tuple[NPImageList, NPImageList, Optional[NPImageList]]:
    '''
    Formats images, prediction masks (index), and optional target masks (index or RGB)
    for segmentation plotting.

    The formatted outputs are used by the segmentation figure functions:
        - `visualize.figures.make_seg_figure_collage`
        - `visualize.figures.make_seg_figure_overlay`
        
    Args:
        imgs (Union[ImageInput, List[ImageInput]]):
            Images to format.
            This supports:
                - A single PIL image
                - A single 3D tensor of shape (channels, height, width)
                - A list of images (each element is a PIL image or 3D tensor)
                - A batched 4D tensor of shape (batch_size, channels, height, width)
        pred_masks (torch.Tensor): 
            Prediction index masks to format.
            This is a 3D tensor of shape (batch_size, height, width)
        idx_to_rgb (optional, Dict[int, RGBTuple]): 
            Dictionary mapping integer indices to RGB tuples.
            This mapping should be one-to-one (injective)
            and the RGB values should be in [0, 255].
        targ_masks (optional, Union[ImageInput, List[ImageInput]]):
            Target masks to format. These may be index or RGB masks.
            This supports:
                - A single PIL image
                - A tensor of a single mask:
                    - RGB mode: shape must be (3, height, width)
                    - Index mode: shape must be (height, width)
                - A list of masks: Each element is a single mask (PIL image or tensor), 
                                   following the rules above.
                - A batched tensor of multiple masks:
                    - RGB mode: shape must be (batch_size, 3, height, width)
                    - Index mode: shape must be (batch_size, height, width)
            This argument is required if `targ_mode` is provided.
        targ_mode (optional, Literal[rgb', 'index']): 
            The mode of the target masks.
                - 'rgb': Target masks are expected to be RGB masks.
                - 'index': Target masks are expected to be index masks.
            This argument is required if `targ_masks` is provided.
        fill_rgb (RGBTuple): 
            RGB tuple used to fill in pixels whose index is not present in `idx_to_rgb`.
            The RGB values should be in [0, 255]. This is only used when `mode='index'`.
            Default is `(114, 114, 114)`.
        visible_rgbs (optional, List[RGBTuple]): 
            List of RGB tuples to set as visible 
            in `pred_masks` and `targ_masks`, after they are represented in RGB.
            The values of each RGB tuple must be in [0, 255].
            If provided, an alpha channel is added, yielding RGBA masks before converting to ndarrays.
            If not provided, RGB masks are directly converted to a ndarray.
    Returns:
        imgs (NPImageList): 
            Images formatted to a list of ndarrays in HWC format.
        pred_masks (NPImageList): 
            Prediction masks formatted to a list of ndarrays in HWC format.
            These are RGB masks (`visible_rgbs` not provided) or RGBA masks (`visible_rgbs` provided).
        targ_masks (NPImageList): 
            Target masks formatted to a list of ndarrays in HWC format.
            These are RGB masks (`visible_rgbs` not provided) or RGBA masks (`visible_rgbs` provided).
    '''
    if not all_or_none(targ_masks, targ_mode):
        raise ValueError(
            'targ_masks and targ_mode must either both be provided '
            'or both be None.'
        )
    
    # Format images
    imgs = format_images(imgs)
    
    # Format prediction masks (index)
    try:
        pred_masks = format_masks(
            pred_masks, 
            mode = 'index', 
            idx_to_rgb = idx_to_rgb,
            fill_rgb = fill_rgb,
            visible_rgbs = visible_rgbs
        )
    except Exception as e:
        raise ValueError('Unable to format prediction masks (pred_masks)') from e
    
    # Format target masks if provided
    if targ_masks is not None:
        try:
            targ_masks = format_masks(
                targ_masks,
                mode = targ_mode,
                idx_to_rgb = idx_to_rgb,
                fill_rgb = fill_rgb,
                visible_rgbs = visible_rgbs
            )
        except Exception as e:
            raise ValueError('Unable to format target masks (targ_masks)') from e

    return imgs, pred_masks, targ_masks


#####################################
# Image Formatting Functions
#####################################
def format_images(imgs: Union[ImageInput, List[ImageInput]]) -> NPImageList:
    '''
    Formats images into a list of ndarrays in HWC format.

    Note: If a PIL image input converts to a 2D ndarray (height, width), 
    a channel dimension to make it 3D (height, width, 1)

    Args:
        imgs (Union[ImageInput, List[ImageInput]]): 
            Images to format.
            This supports:
                - A single PIL image
                - A single 3D tensor of shape (channels, height, width)
                - A list of images (each element is a PIL image or 3D tensor)
                - A batched 4D tensor of shape (batch_size, channels, height, width)

    Returns:
        NPImageList: 
            List of ndarrays converted from `imgs`, all in HWC format.
    '''
    # Single PIL image input
    if isinstance(imgs, Image.Image):
        return [to_numpy_image(imgs)]
    
    # Tensor input
    elif isinstance(imgs, torch.Tensor):
        imgs = ensure_batched(imgs, unbatched_ndim = 3, context_name = 'image tensors')
        return [to_numpy_image(img) for img in imgs]  
    
    # List input
    elif isinstance(imgs, list):
        return [_format_single_image(img) for img in imgs]
    
    else:
        raise TypeError(
            'Expected image samples to be a PIL image, tensor, or list of PIL images/tensors.'
        )
        

def _format_single_image(img: ImageInput) -> np.ndarray:
    '''
    Formats a single image sample into a ndarray in HWC format.

    Note: If a PIL image input converts to a 2D ndarray (height, width), 
    a channel dimension is added to make it 3D (height, width, 1).

    Args:
        img (ImageInput): 
            A single image sample represented as a PIL image or tensor.
            If tensor, must be 3D (channels, height, width).
                      
    Returns:
        np.ndarray: 
            The ndarray converted from `img` in HWC format.
    '''
    if isinstance(img, Image.Image):
        return to_numpy_image(img)
    
    elif isinstance(img, torch.Tensor):
        _validate_ndim(img, ndim = 3, context_name = 'image tensors')
        return to_numpy_image(img)
    
    else:
        raise TypeError(
            f'Expected individual image samples to be PIL images or tensors. '
            f'Got: {type(img)}.'
        )
    

#####################################
# Label Formatting Functions
#####################################
def format_labels(labels: Union[ImageLabel, List[ImageLabel]]) -> IntLabelList:
    '''
    Formats index labels into a list of integer indices.

    Note: Labels must be integer class indices (not one-hot).
    
    Args:
        labels (Union[ImageLabel, List[ImageLabel]]):
            The labels to format.
            This supports:
                - A single integer
                - A single-element tensor
                - A list of labels (each element is an integer or single-element tensor)
                - A batched 1D tensor of shape (batch_size,)
            
    Returns:
        IntLabelList: 
            List of integer indices from `labels`.
    '''
    # Single integer input
    if type(labels) is int:
        return [labels]

    # Tensor input
    elif isinstance(labels, torch.Tensor):
        if labels.numel() == 1:
            return [labels.int().item()]
        elif labels.ndim == 1:
            return labels.int().tolist()
        else:
            raise ValueError(
                'Tensor labels must be single-element or 1D. '
                f'Got a {labels.ndim}D tensor with {labels.numel()} elements.'
            )

    # List input
    elif isinstance(labels, list):
        return [_format_single_label(label) for label in labels]
    
    else:
        raise TypeError(
            'Expected labels to be an integer, tensor, or list of integers/tensors.'
        )


def _format_single_label(label: ImageLabel) -> int:
    '''
    Formats a single index label into an integer index.

    Args:
        label (ImageLabel): 
            Index label represented by an integer or single-element tensor.

    Returns:
        int: 
            The integer index converted from `label`.
    '''
    if type(label) is int:
        return label
    
    elif isinstance(label, torch.Tensor):
        if label.numel() != 1:
            raise ValueError(
                'Expected individual label tensors to be single-element. '
                f'Got a tensor with {label.numel()} elements.'
            )
        return label.int().item()
        
    else:
        raise TypeError(
            'Expected individual labels to be an integer or a tensor. '
            f'Got: {type(label)}'
        )


#####################################
# Mask Formatting Functions
#####################################
def format_masks(
    masks: Union[ImageInput, List[ImageInput]],  
    mode: Literal['rgb', 'index'],
    idx_to_rgb: Optional[Dict[int, RGBTuple]] = None,
    fill_rgb: RGBTuple = (114, 114, 114),
    visible_rgbs: Optional[List[RGBTuple]] = None
) -> NPImageList:
    '''
    Formats segmentation masks into a list of ndarrays.
    Each ndarray is in HWC format and the channel dimension is either size 3 (RGB) or size 4 (RGBA).

    Note: The original mode of PIL image inputs is ignored during formatting.
          If `mode='rgb'`, PIL images should ideally be in RGB mode, as they will always be forced to RGB.
          If `mode='index'`, PIL images should ideally be in grayscale/luminance ('L') mode.
                             They will always be forced to grayscale before mapping to a RGB mask
                             using the provided index-to-color mapping.
                             
    Args:
        masks (Union[ImageInput, List[ImageInput]]):
            The segmentation masks to format.
            This supports:
                - A single PIL image
                - A tensor of a single mask:
                    - RGB mode: shape must be (3, height, width)
                    - Index mode: shape must be (height, width)
                - A list of masks: Each element is a single mask (PIL image or tensor), 
                                   following the rules above.
                - A batched tensor of multiple masks:
                    - RGB mode: shape must be (batch_size, 3, height, width)
                    - Index mode: shape must be (batch_size, height, width)
        mode (Literal[rgb', 'index']): 
            The mode of the input masks.
                - 'rgb': Masks are expected to be RGB masks.
                - 'index': Masks are expected to be index masks.
        idx_to_rgb (optional, Dict[int, RGBTuple]): 
            Dictionary mapping integer indices to RGB tuples.
            This mapping should be one-to-one (injective) and the RGB values should be in [0, 255].
            This argument is required if `mode='index'`.
        fill_rgb (RGBTuple): 
            RGB tuple used to fill in pixels whose index is not present in `idx_to_rgb`.
            The RGB values should be in [0, 255]. 
            This is only used when `mode='index'`.
            Default is (114, 114, 114).
        visible_rgbs (optional, List[RGBTuple]): 
            List of RGB tuples to set as visible in `mask`, after they are represented in RGB.
            The values of each RGB tuple must be in [0, 255].
            If provided, an alpha channel is added, yielding RGBA masks before converting to ndarrays.
            If not provided, the RGB masks are directly converted to a ndarray.
    Returns:
        NPImageList: 
            List of ndarrays, converted from `masks`.
            Each ndarray has shape (height, width, channels).
            If `visible_rgbs` is not provided, channels is 3 (RGB).
            If `visible_rgbs` is provided, channels is 4 (RGBA).
    '''
    if mode == 'rgb':
        return format_rgb_masks(masks, visible_rgbs)
    elif mode == 'index':
        if idx_to_rgb is None:
            raise ValueError('Index-to-RGB mapping must be provided if in index mode')
        return format_idx_masks(masks, idx_to_rgb, fill_rgb, visible_rgbs)
    else:
        raise ValueError("mode must be 'rgb' or 'index'.")


# ---------------------------
# RGB Mask Formatting
# ---------------------------
def format_rgb_masks(
    masks: Union[ImageInput, List[ImageInput]],  
    visible_rgbs: Optional[List[RGBTuple]] = None
) -> NPImageList:
    '''
    Formats segmentation RGB masks into a list of ndarrays.
    Each ndarray is in HWC format and the channel dimension is either size 3 (RGB) or size 4 (RGBA).

    Note: PIL image inputs should ideally be in RGB mode.
          Regardless of the original mode, they will always be forced to RGB.
                             
    Args:
        masks (Union[ImageInput, List[ImageInput]]):
            The RGB masks to format.
            This supports:
                - A single PIL image
                - A 3D tensor of shape (3, height, width)
                - A list of masks: Each element is a single mask (PIL image or tensor), 
                                   following the rules above.
                - A 4D batched tensor of shape (batch_size, 3, height, width)
        visible_rgbs (optional, List[RGBTuple]): 
            List of RGB tuples to set as visible in `masks`.
            The values of each RGB tuple must be in [0, 255].
            If provided, an alpha channel is added, yielding RGBA masks before converting to ndarrays.
            If not provided, `masks` are kept as RGB and are directly converted to ndarrays.

    Returns:
        NPImageList: 
            List of ndarrays, converted from `masks`.
            Each ndarray has shape (height, width, channels).
            If `visible_rgbs` is not provided, channels is 3 (RGB).
            If `visible_rgbs` is provided, channels is 4 (RGBA).
    '''
    # PIL image input
    if isinstance(masks, Image.Image):
        masks = masks.convert('RGB')

        if visible_rgbs is not None:
            masks = F.pil_to_tensor(masks)
            masks = rgb_to_visibility_mask(masks, visible_rgbs) # Shape: (4, height, width)

        return [to_numpy_image(masks)]
    
    # Tensor input
    elif isinstance(masks, torch.Tensor):
        _validate_channel_size(masks, channel_size = 3, context_name = 'RGB masks')
        masks = ensure_batched(masks, unbatched_ndim = 3, context_name = 'RGB masks')

        if visible_rgbs is not None:
            masks = rgb_to_visibility_mask(masks, visible_rgbs) # Shape: (batch_size, 4, height, width)

        return [to_numpy_image(mask) for mask in masks]   
    
    # List input
    elif isinstance(masks, list):
        return [_format_single_rgb_mask(mask, visible_rgbs) for mask in masks]
    
    else:
        raise TypeError(
            'Expected image-like inputs to be a PIL image, tensor, or list of PIL images/tensors.'
        )


def _format_single_rgb_mask(
    mask: ImageInput,   
    visible_rgbs: Optional[List[RGBTuple]] = None
) -> np.ndarray:
    '''
    Formats a single segmentation RGB mask into a ndarray.
    The ndarray is in HWC format and the channel dimension is either size 3 (RGB) or size 4 (RGBA).

    Note: PIL image inputs should ideally be in RGB mode.
          Regardless of the original mode, they will always be forced to RGB.
                             
    Args:
        mask (ImageInput): 
            The RGB mask to format, represented by a PIL image or 3D tensor (3, height, width).
        visible_rgbs (optional, List[RGBTuple]): 
            List of RGB tuples to set as visible in `mask`.
            The values of each RGB tuple must be in [0, 255].
            If provided, an alpha channel is added, yielding a RGBA mask before converting to a ndarray.
            If not provided, `mask` is kept as RGB and is directly converted to a ndarray.

    Returns:
        np.ndarray: 
            The ndarray converted from `masks`. 
            Shape is (height, width, channels).
            If `visible_rgbs` is not provided, channels is 3 (RGB).
            If `visible_rgbs` is provided, channels is 4 (RGBA).
    '''
    # PIL image input
    if isinstance(mask, Image.Image):
        mask = mask.convert('RGB')
        if visible_rgbs is not None:
            mask = F.pil_to_tensor(mask)
            mask = rgb_to_visibility_mask(mask, visible_rgbs) # Shape: (4, height, width)
    
    # Tensor input
    elif isinstance(mask, torch.Tensor):
        _validate_channel_size(mask, channel_size = 3, context_name = 'RGB masks')
        _validate_ndim(mask, ndim = 3, context_name = 'RGB masks')
        if visible_rgbs is not None:
            mask = rgb_to_visibility_mask(mask, visible_rgbs) # Shape: (4, height, width)
    
    else:
        raise TypeError(
            f'Expected individual RGB masks to be PIL images or tensors. '
            f'Got: {type(mask)}.'
        )
    
    return to_numpy_image(mask)
    

# ---------------------------
# Index Mask Formatting
# ---------------------------
def format_idx_masks(
    masks: Union[ImageInput, List[ImageInput]], 
    idx_to_rgb: Dict[int, RGBTuple],
    fill_rgb: RGBTuple = (114, 114, 114),
    visible_rgbs: Optional[List[RGBTuple]] = None
) -> NPImageList:      
    '''
    Formats segmentation index mask into a list of ndarrays.
    Each ndarray is in HWC format and the channel dimension is either size 3 (RGB) or size 4 (RGBA).

    Note: PIL image inputs should ideally be in grayscale/luminance ('L') mode.
          Regardless of the original mode, they will always 
          be forced to grayscale before mapping to a RGB mask
          using the provided index-to-color mapping.
    
    Args:
        masks (Union[ImageInput, List[ImageInput]]): 
            The index masks to format.
            This supports:
                - A single PIL image
                - A 2D tensor of shape (height, width)
                - A list of masks: Each element is a single mask (PIL image or tensor), 
                                   following the rules above.
                - A 3D batched tensor of shape (batch_size, height, width)
        idx_to_rgb (optional, Dict[int, RGBTuple]): 
            Dictionary mapping integer indices to RGB tuples.
            This mapping should be one-to-one (injective) 
            and the RGB values should be in [0, 255].
        fill_rgb (RGBTuple): 
            RGB tuple used to fill in pixels whose index is not present in `idx_to_rgb`.
            The RGB values should be in [0, 255].
            Default is (114, 114, 114).
        visible_rgbs (optional, List[RGBTuple]): 
            List of RGB tuples to set as visible after converting `masks` to RGB masks.
            The values of each RGB tuple must be in [0, 255].
            If provided, an alpha channel is added, yielding RGBA masks before converting to ndarrays.
            If not provided, the RGB masks are directly converted to ndarrays.
                      
    Returns:
        NPImageList: 
            List of ndarrays, converted from `masks`.
            Each ndarray has shape (height, width, channels).
            If `visible_rgbs` is not provided, channels is 3 (RGB).
            If `visible_rgbs` is provided, channels is 4 (RGBA).
    '''
    # PIL image input: Convert to RGB
    if isinstance(masks, Image.Image):
        masks = F.pil_to_tensor(masks.convert('L')) # Shape: (1, height, width)
        masks = idx_to_rgb_mask(masks, idx_to_rgb, fill_rgb) # Shape: (1, 3, height, width)
 
    # Tensor input: Convert to RGB
    elif isinstance(masks, torch.Tensor):
        masks = ensure_batched(masks, unbatched_ndim = 2, context_name = 'index masks')
        masks = idx_to_rgb_mask(masks.long(), idx_to_rgb, fill_rgb) # Shape: (batch_size, 3, height, width)
    
    # List input
    elif isinstance(masks, list):
        return [_format_single_idx_mask(mask, idx_to_rgb, fill_rgb, visible_rgbs) for mask in masks]

    else:
        raise TypeError(
            'Expected mask inputs to be a PIL image, tensor, or list of PIL images/tensors.'
        )

    # PIL image and tensor input: Add alpha channel (if needed) and convert to ndarray
    if visible_rgbs is not None:
        masks = rgb_to_visibility_mask(masks, visible_rgbs) # Shape: (batch_size, 4, height, width)

    return [to_numpy_image(mask) for mask in masks]


def _format_single_idx_mask(
    mask: ImageInput,    
    idx_to_rgb: Dict[RGBTuple, int],
    fill_rgb: RGBTuple = (114, 114, 114),
    visible_rgbs: Optional[List[RGBTuple]] = None
) -> np.ndarray:
    '''
    Formats a single segmentation index mask into a ndarray.
    The ndarray is in HWC format and the channel dimension is either size 3 (RGB) or size 4 (RGBA).

    Note: PIL image inputs should ideally be in grayscale/luminance ('L') mode.
          Regardless of the original mode, they will always 
          be forced to grayscale before mapping to a RGB mask
          using the provided index-to-color mapping.
    
    Args:
        mask (ImageInput): 
            The index mask to format, represented by a PIL image or 2D tensor (height, width).
        idx_to_rgb (optional, Dict[int, RGBTuple]): 
            Dictionary mapping integer indices to RGB tuples.
            This mapping should be one-to-one (injective)
            and the RGB values should be in [0, 255].
        fill_rgb (RGBTuple): 
            RGB tuple used to fill in pixels whose index is not present in `idx_to_rgb`.
            The RGB values should be in [0, 255].
            Default is (114, 114, 114).
        visible_rgbs (optional, List[RGBTuple]): 
            List of RGB tuples to set as visible, after converting `mask` to a RGB mask.
            The values of each RGB tuple must be in [0, 255].
            If provided, an alpha channel is added, yielding a RGBA mask before converting to a ndarray.
            If not provided, the RGB mask is directly converted to a ndarray.
                      
    Returns:
        np.ndarray: 
            The ndarray converted from `masks`. 
            Shape is (height, width, channels).
            If `visible_rgbs` is not provided, channels is 3 (RGB).
            If `visible_rgbs` is provided, channels is 4 (RGBA).
    '''
    # PIL image input
    if isinstance(mask, Image.Image):
        mask = F.pil_to_tensor(mask.convert('L')).squeeze(0) # Shape: (height, width)
    
    # Tensor input
    elif isinstance(mask, torch.Tensor):
        _validate_ndim(mask, ndim = 2, context_name = 'index mask')

    else:
        raise TypeError(
            f'Expected individual index masks to be a PIL image or tensor. '
            f'Got: {type(img)}.'
        )
    
    # Convert to RGB mask
    mask = idx_to_rgb_mask(mask.long(), idx_to_rgb, fill_rgb) # Shape: (3, height, width)

    # Add alpha channel if needed
    if visible_rgbs is not None:
        mask = rgb_to_visibility_mask(mask, visible_rgbs) # Shape: (4, height, width)

    return to_numpy_image(mask)


#####################################
# Helper Functions
#####################################
def to_numpy_image(img: ImageInput) -> np.ndarray:
    '''
    Converts a PIL image or tensor into a ndarray in HWC format.

    Args:
        img (ImageInput): 
            The image to convert, represented as a PIL image or tensor.
                - PIL image: Input is directly converted to a ndarray.
                             If the ndarray is 2D (height, width), 
                             a channel dimension to make it 3D (height, width, 1).

                - Tensor input: If 3D (channels, height, width), it is converted to a 3D ndarray (height, width, channels).
                                If 2D (height, width), it is converted to a 3D ndarray (height, width, 1).
                                If dtype is a sub-dtype of integer, it is converted to `np.uint8`.
                                If dtype is a sub-dtype of float, it is assumed to have values in [0, 1]
                                and is scaled to [0, 255] before converting to `np.uint8`.
    
    Returns:
        np.ndarray: 
            The converted ndarray with shape (height, width, channels).
    '''
    # PIL image input
    if isinstance(img, Image.Image):
        img = np.asarray(img)
        return np.expand_dims(img, 2) if img.ndim == 2 else img
    
    # Tensor input
    img = img.numpy(force = True)

    # Convert to uint8
    img_dtype = img.dtype
    if np.issubdtype(img_dtype, np.integer):
        img = img.astype(np.uint8)

    elif np.issubdtype(img_dtype, np.floating):
        # This assumes the input tensor was scaled to [0, 1]
            # This scales it back to [0, 255] before conversion
        img = (img * 255).astype(np.uint8)

    else:
        raise TypeError(
            f'Got an invalid datatype after numpy conversion: {img_dtype}.'
            'The only supported datatypes are sub-dtypes of np.integer and np.floating.'
        )
    
    # Convert to HWC format
    ndim = img.ndim
    if ndim == 3:
        # Assumes initially in CHW format
        return np.transpose(img, (1, 2, 0)) # Shape: (height, width, channels)
    elif ndim == 2:
        return np.expand_dims(img, 2) # Shape: (height, width, 1)
    else:
        raise ValueError(
            'Expected tensor input to be 3D (channels, height, width) or 2D (height, width). '
            f'Got {ndim} dimensions.'
        )