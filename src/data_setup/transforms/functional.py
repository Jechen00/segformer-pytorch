#####################################
# Imports & Dependencies
#####################################
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

from typing import Sequence, Union, Optional, Tuple

from src.utils import make_tuple, make_range, get_img_size
from src.ml_types import SpatialSize, ImageInput, RGBLike


#####################################
# Functions
#####################################
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
    Functional random affine transformation for a **single** sample.
    This supports transforming both image and optional mask, with separate fill values.
    The same random affine parameters are applied to image and optional mask.

    The transform parameters are sampled using v2.RandomAffine.params: 
        https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomAffine.html

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
                                                           Note that the mask transform always uses `InterpolationMode.NEAREST`.
        img_fill (RGBLike): The fill value for areas outside transformed image, to maintain original shape.
                            This should be a RGB tuple in the same value space as `img`.
                            For example, if `img` is scaled to [0, 1], `img_fill` values should also be scaled to [0, 1].
                            If `int`, assumed `(img_fill, img_fill, img_fill)`.
                            Default is `0`.
        mask_fill (RGBLike): The fill value for areas outside transformed mask, to maintain original shape.
                             This should be a RGB tuple in the same value space as `mask`.
                             For example, if `mask` is scaled to [0, 1], `mask_fill` values should also be scaled to [0, 1].
                             If `int`, assumed `(mask_fill, mask_fill, mask_fill)`.
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
    Functional letterbox transform for a **single** sample.
    This supports transforming both image and optional mask, with separate fill values.
    This is similar to a standard resize transform, 
    but the image is resized to fit within the target dimensions while preserving the aspect ratio. 
    Any remaining space is filled with padding to match the target dimensions.

    Args:
        img (ImageInput):  Input image to transform. If `torch.Tensor`, shape is `(..., height, width)`.
        mask (optional, ImageInput): Segmentation mask for the image, with the same spatial dimensions.
        size (SpatialSize): Size `(height, width)` to transform `img` and (optionally) `mask` into,
                            while preserving their aspect ratios and using padding.
                            If `int`, assumed square.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the image transform.
                                                           Default is `InterpolationMode.BILINEAR`.
                                                           Note that the mask transform always uses `InterpolationMode.NEAREST`.
        img_fill (RGBLike): The fill value to pad transformed image. 
                            This should be a RGB tuple in the same value space as `img`.
                            For example, if `img` is scaled to [0, 1], `img_fill` values should also be scaled to [0, 1].
                            If `int`, assumed `(img_fill, img_fill, img_fill)`.
                            Default is `0`.
        mask_fill (RGBLike): The fill value to pad transformed mask.
                             This should be a RGB tuple in the same value space as `mask`.
                             For example, if `mask` is scaled to [0, 1], `mask_fill` values should also be scaled to [0, 1].
                             If `int`, assumed `(mask_fill, mask_fill, mask_fill)`.
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