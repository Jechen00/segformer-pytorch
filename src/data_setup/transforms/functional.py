#####################################
# Imports & Dependencies
#####################################
from torchvision.transforms import v2
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
import torchvision.transforms.v2.functional as F

from typing import Sequence, Union, Optional, Tuple

from src.utils import make_tuple, make_range, get_img_size
from src.ml_types import SpatialSize, ImageInput, RGBLike


#####################################
# Functions
#####################################
def to_image_and_mask(
    img: ImageInput, 
    mask: Optional[ImageInput] = None
) -> Tuple[tv_tensors.Image, Optional[tv_tensors.Mask]]:
    '''
    Converts an image and optional segmentation mask into `torchvison.tv_tensors`.

    Args:
        img (ImageInput):  Input image to transform. If `torch.Tensor`, shape is `(..., height, width)`.
        mask (optional, ImageInput): Segmentation mask for the image, with the same spatial dimensions.

    Returns:
        img (tv_tensors.Image): The converted image as a `tv_tensors.Image` tensor.
        mask (optional, tv_tensors.Mask): The converted mask as a `tv_tensors.Mask` tensor.
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
    img_fill: RGBLike = 0,
    mask_fill: RGBLike = 255
) -> Tuple[ImageInput, Optional[ImageInput]]:
    '''
    Functional random affine transform for a **single** sample.
    The same random affine parameters are applied to image and optional mask.

    The transform parameters are sampled using v2.RandomAffine.params: 
        https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomAffine.html

    Note: This supports separate fill values for the image and mask.
          The interpolation method for the image is also user-defined,
          while the method for the mask is always `InterpolationMode.NEAREST`.

    Args:
        img (ImageInput):  Input image to transform. If `torch.Tensor`, shape is `(..., height, width)`.
        mask (optional, ImageInput): Segmentation mask for the image, with the same spatial dimensions.
        degrees (Union[float, Sequence[float]]): Range of degrees for rotational transform.
                                                 If `Sequence[float]`, should represent `(min, max)`.
                                                 If `float`, will assume `(-degrees, +degrees`).
                                                 Default is `0.0` for no rotations.
        translate (optional, Sequence[float]): Sequence of the form `(hori_frac, vert_frac)` for translational transforms,
                                               where `hori_frac` and `ver_frac` are the maximum absolute fraction
                                               for horizonal and vertical shifts, respectively.
                                               If `None,` no translations are applied.
        scale (optional, Sequence[float]): Range of factors `(min, max)` for scale transform.
                                           If `None`, no scaling is applied.
        shear (optional, Union[int, float, Sequence[float]]): Range of degrees for shear transform.
                                                              If `Sequence[float]`, should represent `(min_x, max_x)`
                                                              for only x-axis shearing
                                                              or `(min_x, max_x, min_y, max_y)` for x-axis and y-axis shearing.
                                                              If `float`, will assume `(-shear, + shear)`.
                                                              If `None`, no shearing is applied.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the image transform.
                                                           Default is `InterpolationMode.BILINEAR`.
        img_fill (RGBLike): RGB value used to fill areas outside the transformed image, to maintain original shape.
                            This RGB value can be:
                                - a RGB tuple
                                - an integer `x`, assumed to represent `(x, x, x)`.
                            This RGB value should be in the same value space as the expected input images.
                            For example, if the input images are scaled to [0, 1], 
                            `img_fill` values should also be scaled to [0, 1].
                            Default is `0`.
        mask_fill (RGBLike): RGB value used to fill areas outside the transformed mask, to maintain original shape.
                             This RGB value can be:
                                - a RGB tuple
                                - an integer `x`, assumed to represent `(x, x, x)`.
                             This RGB value should be in the same value space as the expected input masks.
                             For example, if the input masks are scaled to [0, 1], 
                             `mask_fill` values should also be scaled to [0, 1].
                             Default is `255`.

    Returns:
        img (ImageInput): Image after applying random affine transform.
                          Spatial size and datatype matches the input image.
        mask (optional, ImageInput): Segmentation mask for the transformed image. 
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


def seg_letterbox(
    img: ImageInput,
    size: SpatialSize, 
    mask: Optional[ImageInput] = None,
    img_interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    img_fill: RGBLike = 0,
    mask_fill: RGBLike = 255
) -> Tuple[ImageInput, Optional[ImageInput]]:
    '''
    Functional letterbox transform for a **single** image and optional segmentation mask.
    This is similar to a standard resize transform,  but the image is resized to fit 
    within the target dimensions while preserving the aspect ratio. 
    Any remaining space is filled with padding to match the target dimensions.

    Note: This supports separate fill values for the image and mask.
          The interpolation method for the image is also user-defined,
          while the method for the mask is always `InterpolationMode.NEAREST`.

    Args:
        img (ImageInput):  Input image to transform. If `torch.Tensor`, shape is `(..., height, width)`.
        mask (optional, ImageInput): Segmentation mask for the image, with the same spatial dimensions.
        size (SpatialSize): Size `(height, width)` to transform `img` and (optionally) `mask` into,
                            while preserving their aspect ratios and using padding.
                            If `int`, assumed square.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the image transform.
                                                           Default is `InterpolationMode.BILINEAR`.
        img_fill (RGBLike): RGB value used to pad transformed image. 
                            This RGB value can be:
                                - a RGB tuple
                                - an integer `x`, assumed to represent `(x, x, x)`.
                            This RGB value should be in the same value space as the expected input images.
                            For example, if the input images are scaled to [0, 1], 
                            `img_fill` values should also be scaled to [0, 1].
                            Default is `0`.
        mask_fill (RGBLike): RGB value used to pad transformed mask.
                             This RGB value can be:
                                - a RGB tuple
                                - an integer `x`, assumed to represent `(x, x, x)`.
                             This RGB value should be in the same value space as the expected input masks.
                             For example, if the input masks are scaled to [0, 1], 
                             `mask_fill` values should also be scaled to [0, 1].
                             Default is `255`.
                             
    Returns:
        img (ImageInput): Image after applying letterbox transform. 
                          Spatial size is `(height, width) = (size[0], size[1])`.
                          Datatype matches the input image.
        
        mask (optional, ImageInput): Segmentation mask for the transformed image. 
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
    pad_l = pad_w // 2
    pad_r = pad_w - pad_l
    pad_t = pad_h // 2
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
    img_interpolation: InterpolationMode = InterpolationMode.BILINEAR
) -> Tuple[ImageInput, Optional[ImageInput]]:
    '''
    Functional resize for a **single** image and optional segmentation mask using `F.resize`.
    This will **not** preserve aspect ratio when resizing.

    Note: The interpolation method for the image is user-defined, 
          while the method for the mask is always `InterpolationMode.NEAREST`.

    Args:
        img (ImageInput):  Input image to transform. If `torch.Tensor`, shape is `(..., height, width)`.
        mask (optional, ImageInput): Segmentation mask for the image, with the same spatial dimensions.
        size (SpatialSize): Size `(height, width)` to transform `img` and (optionally) `mask` into.
                            If `int`, assumed square.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the image transform.
                                                           Default is `InterpolationMode.BILINEAR`.

    Returns:
        img (ImageInput): Image after applying resize transform. 
                          Spatial size is `(height, width) = (size[0], size[1])`.
                          Datatype matches the input image.
        
        mask (optional, ImageInput): Segmentation mask for the resized image. 
                                     Datatype matches the input mask.
                                     This is `None` if an input mask was not provided.
    '''
    img = F.resize(img, size = size, interpolation = img_interpolation)
    if mask is not None:
        mask = F.resize(mask, size = size, interpolation = InterpolationMode.NEAREST)

    return img, mask