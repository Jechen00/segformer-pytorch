#####################################
# Imports & Dependencies
#####################################
import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

from typing import List, Tuple, Literal, Union, Optional, TypeAlias, Sequence

from src.utils import all_or_none
from src.ml_types import FillValue, SpatialSize
from src.data_setup.transforms.ops import (
    ImageTransform, SegRandomAffine, SegLetterbox, 
    SegRandomPerspective, SegResize, ToImageAndMask
)

SizingType: TypeAlias = Literal['letterbox', 'resize']
TransformType: TypeAlias = Literal['phot', 'geo']
GeoTransform: TypeAlias = Union[SegRandomAffine, SegLetterbox, v2.Compose]



#####################################
# Functions
#####################################
def get_base_transforms(
    dtype: torch.dtype = torch.float32,
    scale: bool = True,
    size: Optional[SpatialSize] = None,
    sizing_mode: SizingType = 'letterbox',
    img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    img_fill: FillValue = 0,
    mask_fill: FillValue = 255,
    norm_mean: Optional[Sequence[float]] = None,
    norm_std: Optional[Sequence[float]] = None
) -> v2.Compose:
    '''
    Creates a torchvision pipeline of base transforms.
    This may include:
        1. Optional sizing transforms (resize or letterbox)
        2. Image and Mask tensor conversion (always included)
        3. Datatype conversion and optional scaling to [0, 1] (always included)
        4. Optional Normalization

    Note: datatype conversion and normalization are only applied to images, not segmentation masks.
    
    Args:
        dtype (torch.dtype): The datatype used for datatype conversion. Default is `torch.float32`.
        scale (bool). Whether to scale values to [0, 1] after datatype conversion.
                      This requires `dtype` to be float-like.
                      Default is `True`.
        size (optional, SpatialSize): Size `(height, width)` to resize both image and segmentation mask.
                                      If `int`, size is assumed to be square.
                                      If not provided, no resizing is applied.
        sizing_mode (SizingType): The resizing method to use when `size` is provided.
                                  Supported modes:
                                    - 'letterbox': Uses `SegLetterbox`.
                                                Resizes while preserving aspect ratio and 
                                                applies padding to reach the desired output `size`.
                                    - 'resize': Uses `SegResize`.
                                                Directly scales the image/mask to `size`.
                                                Does not preserve aspect ratio.
                                  Default is `letterbox`.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the sizing transform.
                                                           Default is `InterpolationMode.BILINEAR`.
                                                           Note that the mask transforms always uses `InterpolationMode.NEAREST`.
        img_fill (FillValue): Pixel fill value used for the sizing transform when `sizing_mode='letterbox'`.
                              This can be a float, integer, sequence of floats, or sequence of integers.
                              If scalar (float or integer), the value is used for all channels.
                              If sequence, its length must match the number of channels in the input image.
                              The fill value should be in the same value space as the expected input images.
                              For example, if the input images are scaled to [0, 1], 
                              `img_fill` should also be scaled to [0, 1].
                              Default is `0`.
        mask_fill (FillValue): Pixel fill value used for the sizing transform when `sizing_mode='letterbox'`.
                               This can be a float, integer, sequence of floats, or sequence of integers.
                               If scalar (float or integer), the value is used for all channels.
                               If sequence, its length must match the number of channels in the input mask.
                               The fill value should be in the same value space as the expected input masks.
                               For example, if the input masks are scaled to [0, 1], 
                               `mask_fill` should also be scaled to [0, 1].
                               Default is `255`.
        norm_mean (optional, Sequence[float]): Sequence of means (one for each input channel) used to normalize images.
                                               If provided, `norm_std` must also be provided.
        norm_std (optional, Sequence[float]): Sequence of standard deviations (one for each input channel) used to normalize images.
                                              If provided, `norm_mean` must also be provided.

    Returns:
        v2.Compose: The torchvision pipeline of base transforms.
    '''
    if not all_or_none(norm_mean, norm_std):
        raise ValueError(
            'norm_mean and norm_std must either both be provided or both be None (not provided).'
        )
        
    transforms = []
    if size is not None:
        if sizing_mode == 'letterbox':
            size_transform = SegLetterbox(
                size = size,
                img_interpolation = img_interpolation,
                img_fill = img_fill,
                mask_fill = mask_fill
            )
        elif sizing_mode == 'resize':
            size_transform = SegResize(size = size, img_interpolation = img_interpolation)
        else:
            raise ValueError(f'Unknown sizing mode: {sizing_mode}')
        transforms.append(size_transform)

    transforms.extend([
        ToImageAndMask(),
        v2.ToDtype(dtype = dtype, scale = scale)
    ])

    if norm_mean is not None:
        transforms.append(
            v2.Normalize(mean = norm_mean, std = norm_std)
        )

    return v2.Compose(transforms)


def get_transforms(
    tf_types: Union[TransformType, List[TransformType], Tuple[TransformType, ...]],
    size: Optional[SpatialSize] = None,
    sizing_mode: SizingType = 'letterbox',
    img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    img_fill: FillValue = 0,
    mask_fill: FillValue = 255,
    include_geo_augs: bool = True
) -> Union[ImageTransform, Optional[GeoTransform], v2.Compose]:
    '''
    Creates a torchvision transform pipeline containing photometric and/or geometric transforms.
    Photomeric transforms apply to images only, 
    while geometric transforms are shared between images and optional segmentation masks.

    The photometric transforms are ordered as:
        1) Color Jitter
        2) Random Grayscale (prob = 0.05)
        3) Random Gaussian Blur (prob = 0.1)

    The geometric transforms are ordered as:
        1) Optional resizing (SegResize or SegLetterbox)
        2) Optional geometric augmentations:
            2.1) Random Horizontal Flip (prob = 0.5)
            2.2) Random Affine

    Args:
        tf_types (Union[TransformType, List[TransformType], Tuple[TransformType, ...]]): 
            The types of transforms to include:
                - phot: Photometric transforms.
                - geo: Geometric transfirms.
            To include only one type, input only the string or a singleton list/tuple.
            To include both types, example inputs are `['phot', 'geo']` or `['geo', 'phot']`. 
            The order of the types in `tf_types` determine which transforms are applied first.
        size (optional, SpatialSize): Size `(height, width)` to resize both image and segmentation mask.
                                      If `int`, size is assumed to be square.
                                      If not provided, no resizing is applied.
        sizing_mode (SizingType): The resizing method to use when `size` is provided.
                                  Supported modes:
                                    - 'letterbox': Uses `SegLetterbox`.
                                                Resizes while preserving aspect ratio and 
                                                applies padding to reach the desired output `size`.
                                    - 'resize': Uses `SegResize`.
                                                Directly scales the image/mask to `size`.
                                                Does not preserve aspect ratio.
                                  Default is `letterbox`.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the geometric augmentations of the image.
                                                           Default is `InterpolationMode.BILINEAR`.
                                                           Note that the mask transforms always uses `InterpolationMode.NEAREST`.
        img_fill (FillValue): Pixel fill value used for the image during geometric transforms.
                              This can be a float, integer, sequence of floats, or sequence of integers.
                              If scalar (float or integer), the value is used for all channels.
                              If sequence, its length must match the number of channels in the input image.
                              The fill value should be in the same value space as the expected input images.
                              For example, if the input images are scaled to [0, 1], 
                              `img_fill` should also be scaled to [0, 1].
                              Default is `0`.
        mask_fill (FillValue): Pixel fill value used for the mask during geometric transforms.
                               This can be a float, integer, sequence of floats, or sequence of integers.
                               If scalar (float or integer), the value is used for all channels.
                               If sequence, its length must match the number of channels in the input mask.
                               The fill value should be in the same value space as the expected input masks.
                               For example, if the input masks are scaled to [0, 1], 
                               `mask_fill` should also be scaled to [0, 1].
                               Default is `255`.
        include_geo_augs (bool): Whether to include geometric augmentations in the transform pipeline.
                                 Default is `True`.

    Returns:
        Union[ImageTransform, Optional[GeoTransform], v2.Compose]: 
            Transform pipeline containing photometric and/or geometric augmentations.

            Returns `None` when only geometric transforms are requested (e.g. `tf_types = 'geo'`),
            but no sizing is specified (`size is None`) 
            and geometric augmentations are not included (`include_geo_augs = False`).
    '''
    def get_tf_pipeline(tf_types: TransformType) -> Union[ImageTransform, Optional[GeoTransform]]:
        '''
        Helper that returns the transform pipeline for a single transform type ('phot' or 'geo').
        '''
        if tf_types == 'phot':
            return get_phot_transforms()
        elif tf_types == 'geo':
            return get_geo_transforms(
                size = size,
                sizing_mode = sizing_mode,
                img_interpolation = img_interpolation,
                img_fill = img_fill,
                mask_fill = mask_fill,
                include_augs = include_geo_augs
            ) 
        raise ValueError(f'Unexpected tf_type: {tf_types}')
        
    if isinstance(tf_types, str):
        return get_tf_pipeline(tf_types)
    
    elif isinstance(tf_types, (list, tuple)):
        if len(tf_types) != len(set(tf_types)):
            raise ValueError(
                f'tf_types must not contain duplicates. Got: {tf_types}'
            )
        
        transforms = []
        for tf_type in tf_types:
            tf_pipeline = get_tf_pipeline(tf_type)
            if tf_pipeline is not None:
                transforms.append(tf_pipeline)

        num_transforms = len(transforms)
        if num_transforms == 0:
            return None
        elif num_transforms == 1:
            return transforms[0]
        else:
            return v2.Compose(transforms)
    
    raise TypeError('tf_types must be a string, list, or tuple.')


def get_phot_transforms() -> ImageTransform:
    '''
    Creates an `ImageTransform` pipeline containing only photometric transforms.
    This pipeline only applies transforms to the 'image' of a sample dicationary.

    The transforms are ordered as:
        1) Color Jitter
        2) Random Grayscale (prob = 0.05)
        3) Random Gaussian Blur (prob = 0.1)

    Returns:
        ImageTransform: Pipeline containing photometric transforms.
    '''
    transforms = [
        v2.ColorJitter(
            brightness = 0.5,
            saturation = 0.4,
            hue = 0.02,
            contrast = 0.2
        ),
        v2.RandomGrayscale(p = 0.05),
        v2.RandomApply([
            v2.RandomChoice([
                v2.GaussianBlur(kernel_size = 3, sigma = (0.1, 1.0)),
                v2.RandomAdjustSharpness(sharpness_factor = 2, p = 1)
            ])
        ], p = 0.1)
    ]
    return ImageTransform(transforms)


def get_geo_transforms(
    size: Optional[SpatialSize] = None,
    sizing_mode: SizingType = 'letterbox',
    img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
    img_fill: FillValue = 0,
    mask_fill: FillValue = 255,
    include_augs: bool = True
) -> Optional[GeoTransform]:
    '''
    Creates a torchvision pipeline containing only geometric transforms.
    These transforms are shared between images and their optional segmentation masks.

    The transforms are ordered as:
        1) Optional resizing (SegResize or SegLetterbox)
        2) Optional geometric augmentations:
            2.1) Random Horizontal Flip (prob = 0.5)
            2.2) Random Affine

    Args:
        size (optional, SpatialSize): Size `(height, width)` to resize both image and segmentation mask.
                                      If `int`, size is assumed to be square.
                                      If not provided, no resizing is applied.
        sizing_mode (SizingType): The resizing method to use when `size` is provided.
                                  Supported modes:
                                    - 'letterbox': Uses `SegLetterbox`.
                                                Resizes while preserving aspect ratio and 
                                                applies padding to reach the desired output `size`.
                                    - 'resize': Uses `SegResize`.
                                                Directly scales the image/mask to `size`.
                                                Does not preserve aspect ratio.
                                  Default is `letterbox`.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the transforms of the image.
                                                           Default is `InterpolationMode.BILINEAR`.
                                                           Note that the mask transforms always uses `InterpolationMode.NEAREST`.
        img_fill (FillValue): Pixel fill value used for the image during geometric transforms.
                              This can be a float, integer, sequence of floats, or sequence of integers.
                              If scalar (float or integer), the value is used for all channels.
                              If sequence, its length must match the number of channels in the input image.
                              The fill value should be in the same value space as the expected input images.
                              For example, if the input images are scaled to [0, 1], 
                              `img_fill` should also be scaled to [0, 1].
                              Default is `0`.
        mask_fill (FillValue): Pixel fill value used for the mask during geometric transforms.
                               This can be a float, integer, sequence of floats, or sequence of integers.
                               If scalar (float or integer), the value is used for all channels.
                               If sequence, its length must match the number of channels in the input mask.
                               The fill value should be in the same value space as the expected input masks.
                               For example, if the input masks are scaled to [0, 1], 
                               `mask_fill` should also be scaled to [0, 1].
                               Default is `255`.
        include_augs (bool): Whether to include geometric augmentations in the transform pipeline.
                             Default is `True`.

    Returns:
        optional, GeoTransform: A torchvision transform pipeline containing geometric transforms.

                                Returns a `v2.Compose` pipeline if 
                                geometric augmentations are included (`include_augs = True`).

                                Returns `SegRandomAffine` or `SegLetterbox` if
                                sizing is specified (`size is not None`)
                                and no geometric augmentations are included (`include_augs = False`).

                                Returns `None` when no sizing is specified (`size is None`)
                                and no geometric augmentations are included (`include_augs = False`).
    '''
    if (size is None) and (not include_augs):
        return None
    
    transforms = []
    # Add sizing transform
    if size is not None:
        if sizing_mode == 'letterbox':
            size_transform = SegLetterbox(
                size = size,
                img_interpolation = img_interpolation,
                img_fill = img_fill,
                mask_fill = mask_fill
            )
        elif sizing_mode == 'resize':
            size_transform = SegResize(size = size, img_interpolation = img_interpolation)
        else:
            raise ValueError(f'Unknown sizing mode: {sizing_mode}')
        transforms.append(size_transform)

    # Add geometric transforms
    if include_augs:
        transforms.extend([
            v2.RandomHorizontalFlip(p = 0.5),
            v2.RandomChoice([
                SegRandomAffine(
                    degrees = 5,
                    scale = (0.9, 1.1),
                    translate = (0.05, 0.05),
                    img_interpolation = img_interpolation,
                    img_fill = img_fill,
                    mask_fill = mask_fill
                ),
                SegRandomPerspective(
                    distortion_scale = 0.25,
                    p = 1,
                    img_interpolation = img_interpolation,
                    img_fill = img_fill,
                    mask_fill = mask_fill
                )
            ], p = [0.8, 0.2])
        ])
    return transforms[0] if len(transforms) == 1 else v2.Compose(transforms)