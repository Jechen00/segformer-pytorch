#####################################
# Imports & Dependencies
#####################################
import torch
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

from typing import (
    Sequence, Literal, Union, Optional, 
    Any, Callable, TypeAlias
)
from PIL import Image

from src.utils import make_tuple
from src.ml_types import SpatialSize, ImageInput, RGBLike

AugType: TypeAlias = Literal['phot', 'geo']


#####################################
# Functions
#####################################
def get_base_transforms(
    dtype: torch.dtype = torch.float32, 
    scale: bool = True,
    resize_transform: Optional[Callable] = None
):
    '''
    Creates the torchvision pipeline:
        1) Converts to tv_tensor.Image (v2.ToImage)
        2) Converts data type and optional scaling (v2.ToDtype)
        3) Applies optional resizing (resize_transform)

    Args:
        dtype (torch.dtype): Data type to convert Image tensor into (e.g., torch.float32, torch.float16).
        scale (bool): Whether to scale pixel values to [0, 1].
        resize_transform (optional, Callable): Transform used to resize the image (e.g. v2.Resize).
                                               This should be compatible with v2.Compose.
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
                                                            - 'phot': Photometric augmentations.
                                                            - 'geo': Geometric augmentations.
                                                        To include only one type, input only the string or a singleton list.
                                                        To include both types, an example input is ['phot', 'geo']. 
                                                        Note that photometric augmentations are always 
                                                        applied before geometric augmentations.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the geometric augmentations of the image.
                                                           Default is InterpolationMode.BILINEAR.
                                                           Note that the mask transforms always uses InterpolationMode.NEAREST.
        img_fill (RGBLike): The value used to fill parts of the image during geometric augmentations.
                            This should be a RGB tuple in the same value space as input_dict['image'].
                            For example, if input_dict['image'] is scaled to [0, 1], 
                            img_fill values should also be scaled to [0, 1].
                            If int, assumed (img_fill, img_fill, img_fill).
                            Default is 0.
        mask_fill (RGBLike): The value used to fill parts of the mask during geometric augmentations.
                             This should be a RGB tuple in the same value space as input_dict['image'].
                             For example, if input_dict['image'] is scaled to [0, 1], 
                             mask_fill values should also be scaled to [0, 1].
                             If int, assumed (mask_fill, mask_fill, mask_fill).
                             Default is 255.

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

def functional_seg_letterbox(
    input_dict: dict,
    size: SpatialSize, 
    img_interpolation: InterpolationMode = InterpolationMode.BILINEAR,
    img_fill: RGBLike = 0,
    mask_fill: RGBLike = 255
) -> dict:
    '''
    Functional letterbox transform with support for 
    transforming both image and segmentation masks, with separate fill values.
    This is similar to a standard resize transform, 
    but the image is resized to fit within the target dimensions while preserving the aspect ratio. 
    Any remaining space is filled with padding to match the target dimensions.

    Args:
        input_dict (dict): Input dictionary containing:
                            - image (ImageInput):  Input image to transform. If torch.Tensor, shape is (..., height, width).
                            - mask (optional, ImageInput): Segmentation mask for image, with the same spatial dimensions.
        size (SpatialSize): Size (height, width) to transform img into,
                            while preserving its aspect ratio and using padding.
                            If int, assumed square.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the image transform.
                                                           Default is InterpolationMode.BILINEAR.
                                                           Note that the mask transform always uses InterpolationMode.NEAREST.
        img_fill (RGBLike): The fill value to pad transformed image. 
                            This should be a RGB tuple in the same value space as input_dict['image'].
                            For example, if input_dict['image'] is scaled to [0, 1], 
                            img_fill values should also be scaled to [0, 1].
                            If int, assumed (img_fill, img_fill, img_fill).
                            Default is 0.
        mask_fill (RGBLike): The fill value to pad transformed mask.
                             This should be a RGB tuple in the same value space as input_dict['image'].
                             For example, if input_dict['image'] is scaled to [0, 1], 
                             mask_fill values should also be scaled to [0, 1].
                             If int, assumed (mask_fill, mask_fill, mask_fill).
                             Default is 255.
    Returns:
        dict: Output dictionary containing:
                - image (ImageInput): Image after applying letterbox transform. Shape is (..., size[0], size[1]).
                - mask (optional, ImageInput): Segmentation mask for the transformed image. 
                                                Only exists if a mask was provided in input_dict.
    '''
    output_dict = input_dict.copy()
    img = input_dict['image']
    mask = input_dict.get('mask', None)

    size = make_tuple(size) # (height, width)
    if isinstance(img, torch.Tensor):
        orig_h, orig_w = img.shape[-2:]
    elif isinstance(img, Image.Image):
        orig_w, orig_h = img.size
    else:
        raise TypeError('image in input_dict must be a Tensor or PIL Image.')
        
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
    output_dict['image'] = img

    if mask is not None:
        mask = F.resize(mask, size = (scaled_h, scaled_w), interpolation = InterpolationMode.NEAREST)
        mask = F.pad(mask, padding = (pad_l, pad_t, pad_r, pad_b), fill = mask_fill, padding_mode = 'constant')
        output_dict['mask'] = mask

    return output_dict
    

#####################################
# Classes
#####################################
class SegRandomAffine():
    '''
    Random affine transformation with support for separate fill values for images and segmentation masks.

    The transform parameters are sampled using v2.RandomAffine.params: 
        https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomAffine.html

    Args:
        degrees (Union[float, Sequence]): Range of degrees for rotational transform.
                                          If sequence, should represent (min, max).
                                          If float, will assume (-degrees, +degrees).
        translate (optional, Sequence[float]): Sequence of the form (hori_frac, vert_frac) for translational transforms,
                                               where hori_frac and ver_frac are the maximum absolute fraction
                                               for horizonal and vertical shifts, respectively.
                                               If None, no translations are applied.
        scale (optional, Sequence[float]): Range of factors (min, max) for scale transform.
                                           If None, no scaling is applied.
        shear (optional, Union[int, float, Sequence[float]]): Range of degrees for shear transform.
                                                              If sequence, should represent (min_x, max_x) for only x-axis shearing
                                                              or (min_x, max_x, min_y, max_y) for x-axis and y-axis shearing.
                                                              If float, will assume (-shear, + shear).
                                                              If None, no shearing is applied.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the image transform.
                                                           Default is InterpolationMode.BILINEAR.
                                                           Note that the mask transform always uses InterpolationMode.NEAREST.
        img_fill (RGBLike): The fill value for areas outside transformed image, to maintain original shape.
                            This should be a RGB tuple in the same value space as input_dict['image'].
                            For example, if input_dict['image'] is scaled to [0, 1], 
                            img_fill values should also be scaled to [0, 1].
                            If int, assumed (img_fill, img_fill, img_fill).
                            Default is 0.
        mask_fill (RGBLike): The fill value for areas outside transformed mask, to maintain original shape.
                             This should be a RGB tuple in the same value space as input_dict['image'].
                             For example, if input_dict['image'] is scaled to [0, 1], 
                             mask_fill values should also be scaled to [0, 1].
                             If int, assumed (mask_fill, mask_fill, mask_fill).
                             Default is 255.
    '''
    def __init__(
        self,
        degrees: Union[float, Sequence],
        translate: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None, 
        shear: Optional[Union[int, float, Sequence[float]]] = None, 
        img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        img_fill: RGBLike = 0,
        mask_fill: RGBLike = 255
    ):
        self.degrees = self._to_range(degrees)
        self.translate = translate
        self.scale = scale
        self.shear = self._to_range(shear)
        self.img_interpolation = img_interpolation
        self.img_fill = img_fill
        self.mask_fill = mask_fill

    def __repr__(self) -> str:
        repr_str = (
            f'SegRandomAffine(degrees = {self.degrees}, translate = {self.translate}, '
            f'scale = {self.scale}, shear = {self.shear}, img_interpolation = {self.img_interpolation}, '
            f'img_fill = {self.img_fill}, mask_fill = {self.mask_fill}'
        )
        return repr_str
    
    def __call__(self, input_dict: dict) -> dict:
        '''
        Args:
            input_dict (dict): Input dictionary containing:
                                - image (ImageInput): Input image to transform. If torch.Tensor, shape is (..., height, width).
                                - mask (optional, ImageInput): Segmentation mask for image, with the same spatial dimensions.

        Returns:
            dict: Output dictionary containing:
                    - image (ImageInput): Image after applying transform. Shape is the same as the image in input_dict.
                    - mask (optional, ImageInput): Segmentation mask for the transformed image. 
                                                   Only exists if a mask was provided in input_dict.
        '''
        output_dict = input_dict.copy()
        img = input_dict['image']
        mask = input_dict.get('mask', None)

        if isinstance(img, torch.Tensor):
            img_size = img.shape[-2:]
        elif isinstance(img, Image.Image):
            img_size = img.size[::-1]
        else:
            raise TypeError('image in input_dict must be a Tensor or PIL Image.')

        affine_params = v2.RandomAffine.get_params(
            degrees = self.degrees,
            translate = self.translate,
            scale_ranges = self.scale,
            shears = self.shear,
            img_size = img_size
        )

        output_dict['image'] = F.affine(img, *affine_params, 
                                        interpolation = self.img_interpolation, 
                                        fill = self.img_fill)
        if mask is not None:
            output_dict['mask'] = F.affine(mask, *affine_params, 
                                           interpolation = InterpolationMode.NEAREST, 
                                           fill = self.mask_fill)
        return output_dict
        
    def _to_range(self, x: Optional[Union[int, float, Any]]):
        '''
        Converts an integer or float into the (-x, x), representing a range of values.
        If the input is not an integer or float, then the input is returned unchanged.
        '''
        if isinstance(x, (int, float)):
            return (-x, x)
        else:
            return x
            

