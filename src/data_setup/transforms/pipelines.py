#####################################
# Imports & Dependencies
#####################################
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

from typing import List, Tuple, Literal, Union, Optional, TypeAlias

from src.ml_types import RGBLike, SpatialSize
from src.data_setup.transforms.ops import ImageTransform, SegRandomAffine, SegLetterbox, SegResize

TransformType: TypeAlias = Literal['phot', 'geo']
GeoTransform: TypeAlias = Union[SegRandomAffine, SegLetterbox, v2.Compose]



#####################################
# Functions
#####################################
def get_transforms(
    tf_types: Union[TransformType, List[TransformType], Tuple[TransformType, ...]],
    size: Optional[SpatialSize] = None,
    sizing_mode: Literal['resize', 'letterbox'] = 'letterbox',
    img_interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    img_fill: RGBLike = 0,
    mask_fill: RGBLike = 255,
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
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the geometric augmentations of the image.
                                                           Default is `InterpolationMode.BILINEAR`.
                                                           Note that the mask transforms always uses `InterpolationMode.NEAREST`.
        img_fill (RGBLike): RGB value used to fill parts of the image during geometric transforms.
                            This RGB value can be:
                                - a RGB tuple
                                - an integer `x`, assumed to represent `(x, x, x)`.
                            This RGB value should be in the same value space as the image.
                            For example, if the image is scaled to [0, 1], 
                            `img_fill` values should also be scaled to [0, 1].
                            Default is `0`.
        mask_fill (RGBLike): RGB value used to fill parts of the mask during geometric transforms.
                             This RGB value can be:
                                - a RGB tuple
                                - an integer `x`, assumed to represent `(x, x, x)`.
                             This RGB value should be in the same value space as the segmentation mask.
                             For example, if mask is scaled to [0, 1], 
                             `mask_fill` values should also be scaled to [0, 1].
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
            hue = 0.02
        ),
        v2.RandomGrayscale(p = 0.05),
        v2.RandomApply([v2.GaussianBlur(kernel_size = 3, sigma = (0.1, 1.5))], p = 0.1)
    ]
    return ImageTransform(transforms)


def get_geo_transforms(
    size: Optional[SpatialSize] = None,
    sizing_mode: Literal['resize', 'letterbox'] = 'letterbox',
    img_interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    img_fill: RGBLike = 0,
    mask_fill: RGBLike = 255,
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
        sizing_mode (Literal['resize', 'letterbox']): The resizing method to use when `size` is provided.
                                                      Supported modes:
                                                            - 'resize': Uses `SegResize`.
                                                                        Directly scales the image/mask to `size`.
                                                                        Does not preserve aspect ratio.
                                                            - 'letterbox': Uses `SegLetterbox`.
                                                                        Resizes while preserving aspect ratio and 
                                                                        applies padding to reach the desired output `size`.
                                                      Default is `letterbox`.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the transforms of the image.
                                                           Default is `InterpolationMode.BILINEAR`.
                                                           Note that the mask transforms always uses `InterpolationMode.NEAREST`.
        img_fill (RGBLike): RGB value used to fill parts of the image during geometric transforms.
                            This RGB value can be:
                                - a RGB tuple
                                - an integer `x`, assumed to represent `(x, x, x)`.
                            This RGB value should be in the same value space as the image.
                            For example, if the image is scaled to [0, 1], 
                            `img_fill` values should also be scaled to [0, 1].
                            Default is `0`.
        mask_fill (RGBLike): RGB value used to fill parts of the mask during geometric transforms.
                             This RGB value can be:
                                - a RGB tuple
                                - an integer `x`, assumed to represent `(x, x, x)`.
                             This RGB value should be in the same value space as the segmentation mask.
                             For example, if mask is scaled to [0, 1], 
                             `mask_fill` values should also be scaled to [0, 1].
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
        transforms.append(size_transform)

    # Add geometric transforms
    if include_augs:
        transforms.extend([
            v2.RandomHorizontalFlip(p = 0.5),
            SegRandomAffine(
                degrees = 5,
                scale = (0.9, 1.1),
                translate = (0.05, 0.05),
                img_interpolation = img_interpolation,
                img_fill = img_fill,
                mask_fill = mask_fill
            )
        ])
    return transforms[0] if len(transforms) == 1 else v2.Compose(transforms)