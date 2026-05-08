#####################################
# Imports & Dependencies
#####################################
from torchvision.transforms import InterpolationMode

from abc import ABC, abstractmethod
from typing import (
    Sequence, Union, Optional, Dict, List, Tuple,
    Any, Callable, TypeAlias
)

from src.utils import make_range
from src.ml_types import SpatialSize, RGBLike, ImageInput, SampleDict, SampleListDict
from src.data_setup.transforms import functional

TransformTypes: TypeAlias = Union[Callable, List[Callable], Tuple[Callable, ...]] 


#####################################
# Transform Classes
#####################################
class ImageTransform():
    '''
    Wrapper used to apply a transform **only** to the 'image' key
    of a single-sample or multi-sample dictionary.

    Args:
        transform (Union[Callable, List[Callable]]): A transform or list of transforms to apply only to images.
    '''
    def __init__(self, transforms: TransformTypes):
        self.transforms = transforms

    def __repr__(self) -> str:
        return f'ImageTransform({repr(self._transforms)})'

    def __call__(
        self, 
        input_dict: Union[SampleDict, SampleListDict]
    ) -> Union[SampleDict, SampleListDict]:
        '''
        Applies `self.transforms` to the 'image' key of a single-sample or multi-sample dictionary.

        Args:
            input_dict (Union[SampleDict, SampleListDict]): 
                Input dictionary, with structure depending on whether the input is single or multiple.

                Single-Sample (SampleDict) has the keys (non-exhaustive):
                    - image (ImageInput): Input image to transform. 
                                          If `torch.Tensor`, shape is `(..., height, width)`.

                Multi-Samples (SampleListDict) has the keys (non-exhaustive):
                    - image (List[ImageInput]): List of input images to transform.
                                                If an image is `torch.Tensor`, shape is `(..., height, width)`.

        Returns:
            Union[SampleDict, SampleListDict]: Output dictionary with the same structure as `input_dict`.
                                               The `image` key contains the output after applying `self.transform`.
        '''
        output_dict = input_dict.copy()

        imgs = output_dict['image']
        if isinstance(imgs, list):
            output_dict['image'] = [self._transform_single(img) for img in imgs]
        else:
            output_dict['image'] = self._transform_single(imgs)

        return output_dict
    
    def _transform_single(self, img: ImageInput) -> ImageInput:
        '''
        Applies `self.transforms` to a **single** image.

       Args:
            img (ImageInput): The image to transform. If `torch.Tensor`, shape is `(..., height, width)`.

        Returns:
            ImageInput: Output image after applying `self.transform`.

        '''
        for transform in self._transforms:
            img = transform(img)
        return img

    @property
    def transforms(self) -> TransformTypes:
        return self._transforms
    
    @transforms.setter
    def transforms(self, values: TransformTypes) -> None:
        if isinstance(values, (list, tuple)):
            if not all(callable(val) for val in values):
                raise TypeError(
                    'All elements in transforms must be callable.'
                )
            self._transforms = values

        elif callable(values):
            self._transforms = [values]
            
        else:
            raise TypeError(
                'transforms must be a callable or a list/tuple of callables.'
            )


class SegTransformBase(ABC):
    '''
    Base class for segmentation transforms.
    This is used to support applying a segmentation transform to 
    single-sample (SampleDict) and multi-sample (SampleListDict) input dictionaries.

    Any subclass must implement the `transform_kwargs` property,
    which returns a keyword dictionary containing the arguments needed for `seg_transform`.

    Args:
        seg_transform (Callable): Functional segmentation transform, 
                                  designed to apply to a **single** image and optional mask.
                                  It must accept the arguments `img` and `mask` 
                                  and return a tuple `(img_t, mask_t)`.                
    '''
    def __init__(self, seg_transform: Callable):
        self.seg_transform = seg_transform

    def __call__(
        self, 
        input_dict: Union[SampleDict, SampleListDict]
    ) -> Union[SampleDict, SampleListDict]:
        '''
    Applies `self.seg_transform` to the image(s) and optional mask(s) of an input dictionary.
    Supports both single-sample and multi-sample inputs.
    The structure of the input dictionary is preserved in the output.

        Args:
            input_dict (Union[SampleDict, SampleListDict]): 
                Input dictionary, with structure depending on whether the input is single or multiple.

                Single-Sample (SampleDict) has the keys (non-exhaustive):
                    - image (ImageInput): Input image to transform. 
                                          If `torch.Tensor`, shape is `(..., height, width)`.
                    - mask (optional, ImageInput): Segmentation mask for the image, 
                                                    with the same spatial dimensions.

                Multi-Samples (SampleListDict) has the keys (non-exhaustive):
                    - image (List[ImageInput]): List of input images to transform.
                                                If an image is `torch.Tensor`, 
                                                shape is `(..., height, width)`.
                    - mask (optional, List[ImageInput]): List of the corresponding segmentation masks 
                                                         for the images in `image`,
                                                         with the same spatial dimensions.

        Returns:
            Union[SampleDict, SampleListDict]: Output dictionary with the same structure as `input_dict`.
                                               The `image` and optional `mask` entries contain
                                               the outputs after applying `self.seg_transform`
                                               with arguments from `self.transfrom_kwargs`.
        '''
        output_dict = input_dict.copy()
        imgs = input_dict['image']
        masks = input_dict.get('mask', None)

        no_masks = (masks is None)
        if isinstance(imgs, list):
            if no_masks:
                masks = [None] * len(imgs)
            elif len(imgs) != len(masks):
                raise ValueError(
                    'Image and mask lists in input_dict must be the same length.'
                )
            
            imgs_t, masks_t = [], []
            for img, mask in zip(imgs, masks):
                img, mask = self.seg_transform(img = img, mask = mask, **self.transform_kwargs)
                imgs_t.append(img)
                masks_t.append(mask)
        else:
            imgs_t, masks_t = self.seg_transform(img = imgs, mask = masks, **self.transform_kwargs)

        output_dict['image'] = imgs_t
        if not no_masks:
            output_dict['mask'] = masks_t
        return output_dict
    
    @property
    @abstractmethod
    def transform_kwargs(self) -> Dict[str, Any]:
        '''
        Returns the transform keyword arguments used for `self.seg_transform`.
        '''
        pass
    

class SegRandomAffine(SegTransformBase):
    '''
    Random affine transform with support for separate fill values for images and segmentation masks.
    For a single sample with an image and mask, the same random affine parameters are applied to both.

    This uses `functional.seg_random_affine` to apply transforms.

    Args:
        degrees (Union[float, Sequence[float]]): Range of degrees for rotational transform.
                                                 If `Sequence[float]`, should represent `(min, max)`.
                                                 If `float`, will assume `(-degrees, +degrees`).
                                                 Default is 0.0 for no rotation.
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
                            This should be a RGB tuple in the same value space as the expected input images.
                            For example, if the input images are scaled to [0, 1], 
                            `img_fill` values should also be scaled to [0, 1].
                            If `int`, assumed `(img_fill, img_fill, img_fill)`.
                            Default is `0`.
        mask_fill (RGBLike): The fill value for areas outside transformed mask, to maintain original shape.
                             This should be a RGB tuple in the same value space as the expected input masks.
                             For example, if the input masks are scaled to [0, 1], 
                             `mask_fill` values should also be scaled to [0, 1].
                             If `int`, assumed `(mask_fill, mask_fill, mask_fill)`.
                             Default is `255`.
    '''
    def __init__(
        self,
        degrees: Union[float, Sequence[float]] = 0.0,
        translate: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None, 
        shear: Optional[Union[int, float, Sequence[float]]] = None, 
        img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        img_fill: RGBLike = 0,
        mask_fill: RGBLike = 255
    ):
        super().__init__(seg_transform = functional.seg_random_affine)
        self.degrees = make_range(degrees)
        self.translate = translate
        self.scale = scale
        self.shear = make_range(shear)
        self.img_interpolation = img_interpolation
        self.img_fill = img_fill
        self.mask_fill = mask_fill

    def __repr__(self) -> str:
        '''
        Returns a representational string for the `SegRandomAffine` instance.
        '''
        repr_str = (
            f'SegRandomAffine(degrees = {self.degrees}, translate = {self.translate}, '
            f'scale = {self.scale}, shear = {self.shear}, img_interpolation = {self.img_interpolation}, '
            f'img_fill = {self.img_fill}, mask_fill = {self.mask_fill})'
        )
        return repr_str
        
    @property
    def transform_kwargs(self) -> Dict[str, Any]:
        '''
        Returns the transform keyword arguments used for `functional.seg_random_affine`.
        '''
        return {
            'degrees': self.degrees,
            'translate': self.translate,
            'scale': self.scale,
            'shear': self.shear,
            'img_interpolation': self.img_interpolation,
            'img_fill': self.img_fill,
            'mask_fill': self.mask_fill
        }


class SegLetterbox(SegTransformBase):
    '''
    Letterbox transform that supports transforming both image and optional mask, with separate fill values.
    This is similar to a standard resize transform, 
    but the image is resized to fit within the target dimensions while preserving the aspect ratio. 
    Any remaining space is filled with padding to match the target dimensions.

    This uses `functional.seg_letterbox` to apply transforms.

    Args:
        size (SpatialSize): Size `(height, width)` to transform the image and optional mask into,
                            while preserving their aspect ratios and using padding.
                            If `int`, assumed square.
        img_interpolation (Union[InterpolationMode, int]): Interpolation mode used for the image transform.
                                                           Default is `InterpolationMode.BILINEAR`.
                                                           Note that the mask transform always uses `InterpolationMode.NEAREST`.
        img_fill (RGBLike): The fill value to pad transformed image. 
                            This should be a RGB tuple in the same value space as the expected input images.
                            For example, if the input images are scaled to [0, 1], 
                            `img_fill` values should also be scaled to [0, 1].
                            If `int`, assumed `(img_fill, img_fill, img_fill)`.
                            Default is `0`.
        mask_fill (RGBLike): The fill value to pad transformed mask.
                             This should be a RGB tuple in the same value space as the expected input masks.
                             For example, if the input masks are scaled to [0, 1], 
                             `mask_fill` values should also be scaled to [0, 1].
                             If `int`, assumed `(mask_fill, mask_fill, mask_fill)`.
                             Default is `255`.
    '''
    def __init__(
        self,
        size: SpatialSize, 
        img_interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        img_fill: RGBLike = 0,
        mask_fill: RGBLike = 255
    ):
        super().__init__(seg_transform = functional.seg_letterbox)
        self.size = size
        self.img_interpolation = img_interpolation
        self.img_fill = img_fill
        self.mask_fill = mask_fill
    
    def __repr__(self) -> str:
        '''
        Returns a representational string for the `SegRandomAffine` instance.
        '''
        repr_str = (
            f'SegLetterBox(size = {self.size}, img_interpolation = {self.img_interpolation}, '
            f'img_fill = {self.img_fill}, mask_fill = {self.mask_fill})'
        )
        return repr_str

    @property
    def transform_kwargs(self) -> Dict[str, Any]:
        '''
        Returns the transform keyword arguments used for `functional.seg_letterbox`.
        '''
        return {
            'size': self.size,
            'img_interpolation': self.img_interpolation,
            'img_fill': self.img_fill,
            'mask_fill': self.mask_fill
        }