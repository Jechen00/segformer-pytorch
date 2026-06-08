#####################################
# Imports & Dependencies
#####################################
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2.functional as F

import cv2
import numpy as np
import math

from typing import Sequence, Union, Optional, Tuple, Literal

from src.utils.data_utils import make_tuple, make_range, get_img_size
from src.ml_types import SpatialSize, ImageInput, FillValue


#####################################
# Functions
#####################################
def to_image_and_mask(
    img: ImageInput, 
    mask: Optional[ImageInput] = None
) -> Tuple[tv_tensors.Image, Optional[tv_tensors.Mask]]:
    '''
    Converts an image and optional segmentation mask into `torchvison.tv_tensors`.

    Note: Supported datatypes for the image and mask are:
        - PIL image
        - tensor of shape (..., height, width)

    Args:
        img (ImageInput):  
            Input image to transform. 
        mask (optional, ImageInput): 
            Segmentation mask for the image, with the same spatial dimensions.

    Returns:
        img (tv_tensors.Image): 
            The converted image as a `tv_tensors.Image` tensor.
        mask (optional, tv_tensors.Mask): 
            The converted mask as a `tv_tensors.Mask` tensor.
            This is `None` if an input mask was not provided.
    '''
    img = tv_tensors.Image(img)
    mask = tv_tensors.Mask(mask) if mask is not None else None
    return img, mask


def seg_random_affine(
    img: ImageInput,
    mask: Optional[ImageInput] = None,
    degrees: Union[float, Sequence[float]] = 0.0,
    translate: Optional[Sequence[float]] = None,
    scale: Optional[Sequence[float]] = None, 
    shear: Optional[Union[int, float, Sequence[float]]] = None, 
    img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    img_fill: FillValue = 0,
    mask_fill: FillValue = 255
) -> Tuple[ImageInput, Optional[ImageInput]]:
    '''
    Functional random affine transform for a **single** sample.
    The same random affine parameters are applied to image and optional mask.

    Note: This supports separate fill values for the image and mask.
          The interpolation method for the image is also user-defined,
          while the method for the mask is always `InterpolationMode.NEAREST`.
    
    Note: Supported datatypes for the image and mask are:
        - PIL image
        - tensor of shape (..., height, width)

    Note: The transform parameters are sampled using v2.RandomAffine.get_params: 
        https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomAffine.html

    Args:
        img (ImageInput):  
            Input image to transform. 
        mask (optional, ImageInput):    
            Segmentation mask for the image, with the same spatial dimensions.
        degrees (Union[float, Sequence[float]]): 
            Range of degrees for rotational transform.
            If `Sequence[float]`, should represent `(min, max)`.
            If `float`, will assume `(-degrees, +degrees`).
            Default is `0.0` for no rotations.
        translate (optional, Sequence[float]): 
            Sequence of the form `(hori_frac, vert_frac)` for translational transforms,
            where `hori_frac` and `ver_frac` are the maximum absolute fraction
            for horizonal and vertical shifts, respectively.
            If `None,` no translations are applied.
        scale (optional, Sequence[float]): 
            Range of factors `(min, max)` for scale transform.
            If `None`, no scaling is applied.
        shear (optional, Union[int, float, Sequence[float]]): 
            Range of degrees for shear transform.
            If `Sequence[float]`, should represent `(min_x, max_x)` for only x-axis shearing
            or `(min_x, max_x, min_y, max_y)` for x-axis and y-axis shearing.
            If `float`, will assume `(-shear, + shear)`.
            If `None`, no shearing is applied.
        img_interpolation (Union[InterpolationMode, int]): 
            Interpolation mode used for the image transform.
            Default is `InterpolationMode.BILINEAR`.
        img_fill (FillValue): 
            Pixel fill value used for areas outside the transformed image, to maintain original shape.
            This can be a float, integer, sequence of floats, or sequence of integers.
            If scalar (float or integer), the value is used for all channels.
            If sequence, its length must match the number of channels in the input image.
            The fill value should be in the same value space as the expected input images.
            For example, if the input images are scaled to [0, 1], 
            `img_fill` should also be scaled to [0, 1].
            Default is `0`.
        mask_fill (FillValue): 
            Pixel fill value used for areas outside the transformed mask, to maintain original shape.
            This can be a float, integer, sequence of floats, or sequence of integers.
            If scalar (float or integer), the value is used for all channels.
            If sequence, its length must match the number of channels in the input mask.
            The fill value should be in the same value space as the expected input masks.
            For example, if the input masks are scaled to [0, 1], 
            `mask_fill` should also be scaled to [0, 1].
            Default is `255`.

    Returns:
        img (ImageInput): 
            Image after applying random affine transform.
            Spatial size and datatype matches the input image.
        mask (optional, ImageInput): 
            Segmentation mask for the transformed image. 
            Spatial size and datatype matches the input mask.
            This is `None` if an input mask was not provided.
    '''
    img_size = get_img_size(img)
    
    # Get a single set of random affine parameters to apply to both image and mask
    affine_params = v2.RandomAffine.get_params(
        degrees = make_range(degrees),
        translate = translate,
        scale_ranges = scale,
        shears = make_range(shear),
        img_size = img_size
    )

    # Apply affine transform
    img = F.affine(img, *affine_params, 
                   interpolation = img_interpolation, 
                   fill = img_fill)
    if mask is not None:
        mask = F.affine(mask, *affine_params, 
                        interpolation = InterpolationMode.NEAREST, 
                        fill = mask_fill)
    return img, mask


def seg_random_perspective(
    img: ImageInput, 
    mask: Optional[ImageInput] = None,
    distortion_scale: float = 0.5, 
    p: float = 0.5,
    img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    img_fill: FillValue = 0,
    mask_fill: FillValue = 255
) -> Tuple[ImageInput, ImageInput]:
    '''
    Functional random perspective transform for a **single** sample.
    The same random perspective parameters are applied to image and optional mask.

    Note: This supports separate fill values for the image and mask.
          The interpolation method for the image is also user-defined,
          while the method for the mask is always `InterpolationMode.NEAREST`.
    
    Note: Supported datatypes for the image and mask are:
        - PIL image
        - tensor of shape (..., height, width)

    Note: The transform parameters are sampled using v2.RandomPerspective.get_params: 
        https://docs.pytorch.org/vision/main/generated/torchvision.transforms.RandomPerspective.html

    Args:
        img (ImageInput):  
            Input image to transform. 
        mask (optional, ImageInput): 
            Segmentation mask for the image, with the same spatial dimensions.
        distortion_scale (float): 
            Value to control the degree of distortion from the random perspective transform.
            Must be in the range `[0, 1]`. Default is `0.5`.
        p (float): 
            Probability of applying the random perspective transform to `img` and `mask`.
            Default is `0.5`.
        img_interpolation (Union[InterpolationMode, int]): 
            Interpolation mode used for the image transform.
            Default is `InterpolationMode.BILINEAR`.
        img_fill (FillValue): 
            Pixel fill value used for areas outside the transformed image, to maintain original shape.
            This can be a float, integer, sequence of floats, or sequence of integers.
            If scalar (float or integer), the value is used for all channels.
            If sequence, its length must match the number of channels in the input image.
            The fill value should be in the same value space as the expected input images.
            For example, if the input images are scaled to [0, 1], 
            `img_fill` should also be scaled to [0, 1].
            Default is `0`.
        mask_fill (FillValue): 
            Pixel fill value used for areas outside the transformed mask, to maintain original shape.
            This can be a float, integer, sequence of floats, or sequence of integers.
            If scalar (float or integer), the value is used for all channels.
            If sequence, its length must match the number of channels in the input mask.
            The fill value should be in the same value space as the expected input masks.
            For example, if the input masks are scaled to [0, 1], 
            `mask_fill` should also be scaled to [0, 1].
            Default is `255`.

    Returns:
        img (ImageInput): 
            Output image.
            With probability `p`, the random perspective transform is applied.
            With probability `1-p`, the original input image is returned.
            The spatial size and datatype always matches the input image.
        mask (optional, ImageInput): 
            Segmentation mask for the output image. 
            Spatial size and datatype matches the input mask.
            This is `None` if an input mask was not provided.
    '''
    if torch.rand(1) < p:
        # Get a single set of random perspective parameters to apply to both image and mask
        h, w = get_img_size(img)
        startpoints, endpoints = v2.RandomPerspective.get_params(w, h, distortion_scale)

        # Apply perspective transform
        img = F.perspective(img, startpoints, endpoints, 
                            interpolation = img_interpolation, 
                            fill = img_fill)
        if mask is not None:
            mask = F.perspective(mask, startpoints, endpoints, 
                                 interpolation = InterpolationMode.NEAREST, 
                                 fill = mask_fill)
    return img, mask


def seg_letterbox(
    img: ImageInput,
    size: SpatialSize, 
    mask: Optional[ImageInput] = None,
    img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    img_fill: FillValue = 0,
    mask_fill: FillValue = 255
) -> Tuple[ImageInput, Optional[ImageInput]]:
    '''
    Functional letterbox transform for a **single** image and optional segmentation mask.
    This is similar to a standard resize transform,  but the image is resized to fit 
    within the target dimensions while preserving the aspect ratio. 
    Any remaining space is filled with padding to match the target dimensions.

    Note: This supports separate fill values for the image and mask.
          The interpolation method for the image is also user-defined,
          while the method for the mask is always `InterpolationMode.NEAREST`.

    Note: Supported datatypes for the image and mask are:
        - PIL image
        - tensor of shape (..., height, width)

    Args:
        img (ImageInput):  
            Input image to transform. 
        mask (optional, ImageInput): 
            Segmentation mask for the image, with the same spatial dimensions.
        size (SpatialSize): 
            Size `(height, width)` to transform `img` and (optionally) `mask` into,
            while preserving their aspect ratios and using padding.
            If `int`, assumed square.
        img_interpolation (Union[InterpolationMode, int]): 
            Interpolation mode used for the image transform.
            Default is `InterpolationMode.BILINEAR`.
        img_fill (FillValue): 
            Pixel fill value used to pad the transformed image. 
            This can be a float, integer, sequence of floats, or sequence of integers.
            If scalar (float or integer), the value is used for all channels.
            If sequence, its length must match the number of channels in the input image.
            The fill value should be in the same value space as the expected input images.
            For example, if the input images are scaled to [0, 1], 
            `img_fill` should also be scaled to [0, 1].
            Default is `0`.
        mask_fill (FillValue): 
            Pixel fill value used to pad the transformed mask. 
            This can be a float, integer, sequence of floats, or sequence of integers.
            If scalar (float or integer), the value is used for all channels.
            If sequence, its length must match the number of channels in the input mask.
            The fill value should be in the same value space as the expected input masks.
            For example, if the input masks are scaled to [0, 1], 
            `mask_fill` should also be scaled to [0, 1].
            Default is `255`.
                               
    Returns:
        img (ImageInput): 
            Image after applying letterbox transform. 
            Spatial size is `(height, width) = (size[0], size[1])`.
            Datatype matches the input image.
        
        mask (optional, ImageInput): 
            Segmentation mask for the transformed image. 
            Datatype matches the input mask.
            This is `None` if an input mask was not provided.
    '''
    size = make_tuple(size) # (height, width)
    orig_h, orig_w = get_img_size(img)
        
    # Resizing
    lb_scale = min(size[1] / orig_w, size[0] / orig_h)
    scaled_w = int(orig_w * lb_scale)
    scaled_h = int(orig_h * lb_scale)
    
    # Padding
    pad_w = size[1] - scaled_w
    pad_h = size[0] - scaled_h
    pad_l = math.ceil(pad_w / 2)
    pad_r = pad_w - pad_l
    pad_t = math.ceil(pad_h / 2)
    pad_b = pad_h - pad_t
    
    # Applying Transforms
    img = F.resize(img, size = (scaled_h, scaled_w), interpolation = img_interpolation)
    img = F.pad(img, padding = (pad_l, pad_t, pad_r, pad_b), fill = img_fill, padding_mode = 'constant')

    if mask is not None:
        mask = F.resize(mask, size = (scaled_h, scaled_w), interpolation = InterpolationMode.NEAREST)
        mask = F.pad(mask, padding = (pad_l, pad_t, pad_r, pad_b), fill = mask_fill, padding_mode = 'constant')

    return img, mask


def seg_resize(
    img: ImageInput, 
    size: SpatialSize,
    mask: Optional[ImageInput] = None,    
    img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR
) -> Tuple[ImageInput, Optional[ImageInput]]:
    '''
    Functional resize for a **single** image and optional segmentation mask using `F.resize`.
    This will **not** preserve aspect ratio when resizing.

    Note: The interpolation method for the image is user-defined, 
          while the method for the mask is always `InterpolationMode.NEAREST`.

    Note: Supported datatypes for the image and mask are:
        - PIL image
        - tensor of shape (..., height, width)

    Args:
        img (ImageInput):  
            Input image to transform. 
        mask (optional, ImageInput): 
            Segmentation mask for the image, with the same spatial dimensions.
        size (SpatialSize): 
            Size `(height, width)` to transform `img` and (optionally) `mask` into.
            If `int`, assumed square.
        img_interpolation (Union[InterpolationMode, int]): 
            Interpolation mode used for the image transform.
            Default is `InterpolationMode.BILINEAR`.

    Returns:
        img (ImageInput): 
            Image after applying resize transform. 
            Spatial size is `(height, width) = (size[0], size[1])`.
            Datatype matches the input image.
        
        mask (optional, ImageInput): 
            Segmentation mask for the resized image. 
            Datatype matches the input mask.
            This is `None` if an input mask was not provided.
    '''
    img = F.resize(img, size = size, interpolation = img_interpolation)
    if mask is not None:
        mask = F.resize(mask, size = size, interpolation = InterpolationMode.NEAREST)

    return img, mask


def reverse_letterbox(
    img: ImageInput,
    orig_size: Tuple[int, int],
    resize_to_orig: bool = True,
    interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR
) -> ImageInput:
    '''
    Reverses the `seg_letterbox` transform by removing the padding 
    and, optionally, resizing back to the original image size.

    Note: Supported datatypes for the image input is:
        - PIL image
        - tensor of shape (..., height, width)

    Args:
        img (ImageInput): 
            The input image that previously passed through the `seg_letterbox` transform.
        orig_size (Tuple[int, int]): 
            The original size (height, width) of `img` before the letterbox transform was applied.
        resize_to_orig (bool): 
            Whether to resize `img` back to `orig_size` after removing letterbox padding.
            If False, only the padding is removed, and the returned image stays at the letterboxed scale.
            Default is `True`.
        interpolation (Union[InterpolationMode, int]): 
            The interpolation mode to use when resizing `img` back to `orig_size` (after padding is removed).
            This is only used when `resize_to_orig = True`.
            If `img` is a segmentation mask, it is recommended to use `InterpolationMode.NEAREST`.
            Default is `InterpolationMode.BILINEAR`.

    Returns:
        ImageInput: 
            The input image with letterbox padding removed 
            and, optionally, resized back to its original size.
            Datatype matches `img`.
    '''
    orig_h, orig_w = orig_size
    lb_h, lb_w = get_img_size(img)
    
    lb_scale = min(lb_h/orig_h, lb_w/orig_w)
    scaled_h = int(orig_h * lb_scale)
    scaled_w = int(orig_w * lb_scale)
    
    # Remove padding
    pad_rm_img = F.center_crop(img, output_size = (scaled_h, scaled_w))
    
    # Resize back to original size if needed
    if resize_to_orig:
        return F.resize(pad_rm_img, size = orig_size, interpolation = interpolation)
    else:
        return pad_rm_img
    

def reverse_letterbox_numpy(
    img: np.ndarray,
    orig_size: Tuple[int, int],
    resize_to_orig: bool = True,
    interpolation: int = cv2.INTER_LINEAR,
    input_format: Literal['HWC', 'CHW'] = 'HWC'
) -> np.ndarray:
    '''
    Reverses the `seg_letterbox` transform applied to an image that has been converted to a ndarray.
    This is done by removing the padding and, optionally, resizing back to the original image size.

    Args:
        img (np.ndarray): 
            The input image represented as a ndarray.
            This image should have previously passed through a `seg_letterbox` transform.                 
        orig_size (Tuple[int, int]): 
            The original size (height, width) of `img` before the letterbox transform was applied.
        resize_to_orig (bool): 
            Whether to resize `img` back to `orig_size` after removing letterbox padding.
            If False, only the padding is removed, and the returned image stays at the letterboxed scale.
            Default is `True`.
        interpolation (int): 
            The interpolation mode to use when resizing `img` back to `orig_size` (after padding is removed).
            This should be an OpenCV interpolation flag (e.g. `cv2.INTER_LINEAR`).
            This is only used when `resize_to_orig = True`.
            If `img` is a segmentation mask, it is recommended 
            to use `cv2.INTER_NEAREST_EXACT` or `cv2.INTER_NEAREST` for nearest interpolation.
            Default is `cv2.INTER_LINEAR` for bilinear interpolation.

    Returns:
        np.ndarray: 
            The input image with letterbox padding removed 
            and, optionally, resized back to its original size.
    '''
    # Format to HWC
    if input_format == 'CHW':
        img = img.transpose(1, 2, 0)

    # Numpy center crop to (scaled_h, scaled_w)
    orig_h, orig_w = orig_size
    lb_h, lb_w = img.shape[:2]
    
    lb_scale = min(lb_h/orig_h, lb_w/orig_w)
    scaled_h = int(orig_h * lb_scale)
    scaled_w = int(orig_w * lb_scale)

    pad_t = math.ceil((lb_h - scaled_h) / 2)
    pad_l = math.ceil((lb_w - scaled_w) / 2)

    img = img[pad_t:(pad_t + scaled_h), 
              pad_l:(pad_l + scaled_w)]
    
    # OpenCV resize to original (if needed)
    if resize_to_orig:
        # OpenCV assumes HWC input shape, but dsize is (resized_w, resized_h)
        # Output shape: (orig_h, orig_w, channels)
        img = cv2.resize(img, dsize = (orig_w, orig_h), interpolation = interpolation)

    # Format back to input format
    if input_format == 'CHW':
        img = img.transpose(1, 2, 0)

    return img