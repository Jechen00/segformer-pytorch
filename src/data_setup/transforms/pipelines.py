#####################################
# Imports & Dependencies
#####################################
import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode

from typing import Sequence, Literal, Union, Optional, Callable, TypeAlias

from src.ml_types import RGBLike
from src.data_setup.transforms.ops import SegRandomAffine

AugType: TypeAlias = Literal['phot', 'geo']


#####################################
# Compose Functions
#####################################
def get_base_transforms(
    dtype: torch.dtype = torch.float32, 
    scale: bool = True,
    resize_transform: Optional[Callable] = None
) -> v2.Compose:
    '''
    Creates the torchvision pipeline:
        1) Converts to `tv_tensor.Image` (`v2.ToImage`)
        2) Converts data type and optional scaling (`v2.ToDtype`)
        3) Applies optional resizing (`resize_transform`)

    Args:
        dtype (torch.dtype): Data type to convert Image tensor into (e.g., `torch.float32`, `torch.float16`).
        scale (bool): Whether to scale pixel values to [0, 1].
        resize_transform (Optional[Callable]): Transform used to resize the image (e.g. `v2.Resize`).
                                               This should be compatible with `v2.Compose`.
    Returns:
        v2.Compose: Torchvision pipeline for converting to tensor and optional resizing.
    '''
    transforms = [
        v2.ToImage(),
        v2.ToDtype(dtype, scale = scale)
    ]
    if resize_transform is not None:
        transforms.append(resize_transform)
    return v2.Compose(transforms)


def get_augmentations(
    aug_types: Union[AugType, Sequence[AugType]],
    img_interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    img_fill: RGBLike = 0,
    mask_fill: RGBLike = 255
) -> v2.Compose:
    '''
    Creates a torchvision transform pipeline containing only data augmentations.
    These transforms support both image classification and semantic segmentation.
    They may include photometric and/or geometric augmentations.

    The photometric augmentations are ordered as:
        1) Color Jitter
        2) Random Grayscale (prob = 0.05)
        3) Random Gaussian Blur (prob = 0.1)

    The geometric augmentations are ordered as:
        1) Random Horizontal Flip (prob = 0.5)
        2) Random Affine

    Args:
        aug_types (Union[AugType, Sequence[AugType]]): The types of augmentations to include:
                                                            - phot: Photometric augmentations.
                                                            - geo: Geometric augmentations.
                                                        To include only one type, input only the string or a singleton list.
                                                        To include both types, an example input is `['phot', 'geo']`. 
                                                        Note that photometric augmentations are always 
                                                        applied before geometric augmentations.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the geometric augmentations of the image.
                                                           Default is `InterpolationMode.BILINEAR`.
                                                           Note that the mask transforms always uses `InterpolationMode.NEAREST`.
        img_fill (RGBLike): The value used to fill parts of the image during geometric augmentations.
                            This should be a RGB tuple in the same value space as `input_dict['image']`.
                            For example, if `input_dict['image']` is scaled to [0, 1], 
                            `img_fill` values should also be scaled to [0, 1].
                            If `int`, assumed `(img_fill, img_fill, img_fill)`.
                            Default is `0`.
        mask_fill (RGBLike): The value used to fill parts of the mask during geometric augmentations.
                             This should be a RGB tuple in the same value space as `input_dict['image']`.
                             For example, if `input_dict['image']` is scaled to [0, 1], 
                             `mask_fill` values should also be scaled to [0, 1].
                             If `int`, assumed `(mask_fill, mask_fill, mask_fill)`.
                             Default is `255`.

    Returns:
        v2.Compose: A torchvision transform pipeline containing photometric and/or geometric augmentations.
    '''
    if isinstance(aug_types, str):
        aug_types = [aug_types]
    
    if (not aug_types) or (not set(aug_types).issubset({'phot', 'geo'})):
        raise ValueError(
            "aug_types must be a non-empty sequence where each element is either 'phot' or 'geo'."
        )
        
    transforms = []
    # Add photometric transforms
    if 'phot' in aug_types:
        transforms.extend([
            v2.ColorJitter(
                brightness = 0.5,
                saturation = 0.4,
                hue = 0.02
            ),
            v2.RandomGrayscale(p = 0.05),
            v2.RandomApply([v2.GaussianBlur(kernel_size = 3, sigma = (0.1, 1.5))], p = 0.1)
        ])
        
    # Add geometrix transforms
    if 'geo' in aug_types:
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
    return v2.Compose(transforms)
            

