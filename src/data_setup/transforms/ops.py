#####################################
# Imports & Dependencies
#####################################
from torchvision.transforms import v2, InterpolationMode

from abc import ABC, abstractmethod
from typing import (
    Sequence, Union, Optional, Dict, List, Tuple,
    Any, Callable, TypeAlias
)

from src.utils import make_range
from src.data_setup.transforms import functional
from src.data_setup.types import SampleDict, SampleListDict
from src.ml_types import SpatialSize, FillValue, ImageInput

TransformLike: TypeAlias = Union[Callable, List[Callable], Tuple[Callable, ...]] 


#####################################
# Image-Only Transform Classes
#####################################
class ImageTransform():
    '''
    Wrapper used to apply a transform **only** to the 'image' key
    of a single-sample or multi-sample dictionary.

    Args:
        transform (Union[Callable, List[Callable]]): 
            A transform or list of transforms to apply only to images.
    '''
    def __init__(self, transforms: TransformLike):
        self.transforms = transforms

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({repr(self.transforms)})'

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

        Returns:i
            Union[SampleDict, SampleListDict]: 
                Output dictionary with the same structure as `input_dict`.
                The `image` key contains the output after applying `self.transform`.
        '''
        output_dict = input_dict.copy()

        imgs = output_dict['image']
        if isinstance(imgs, list):
            output_dict['image'] = [self._transforms(img) for img in imgs]
        else:
            output_dict['image'] = self._transforms(imgs)

        return output_dict

    @property
    def transforms(self) -> Union[Callable, v2.Compose]:
        return self._transforms
    
    @transforms.setter
    def transforms(self, values: TransformLike) -> None:
        if isinstance(values, (list, tuple)):
            num_transforms = len(values)
            if num_transforms == 0:
                raise ValueError('transforms cannot be empty.')
            if not all(callable(v) for v in values):
                raise ValueError('All elements in transforms must be callable')

            if num_transforms == 1:
                    self._transforms = values[0]
            else:
                self._transforms = v2.Compose(values)

        elif callable(values):
            self._transforms = values
            
        else:
            raise TypeError(
                'transforms must be a callable or a non-empty list/tuple of callables.'
            )


#####################################
# Segmentation-Supported Classes
#####################################
class SegTransformBase(ABC):
    '''
    Base class for segmentation transforms.
    This is used to support applying a segmentation transform to 
    single-sample (SampleDict) and multi-sample (SampleListDict) input dictionaries.

    Any subclass must implement the `transform_kwargs` property,
    which returns a keyword dictionary containing the arguments needed for `seg_transform`.

    Args:
        seg_transform (Callable): 
            Functional segmentation transform, designed to apply to a **single** image and optional mask.
            It must accept the arguments `img` and `mask` and return a tuple `(img_t, mask_t)`.                
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
            Union[SampleDict, SampleListDict]: 
                Output dictionary with the same structure as `input_dict`.
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
    

class ToImageAndMask(SegTransformBase):
    '''
    Converts image(s) and optional segmentation mask(s) into `torchvison.tv_tensors`.

    This uses `functional.to_image_and_mask` to apply transforms.
    '''
    def __init__(self):
        super().__init__(seg_transform = functional.to_image_and_mask)
    
    def __repr__(self) -> str:
        '''
        Returns a representational string for the `ToImageAndMask` instance.
        '''
        return 'ToImageAndMask()'
    
    @property
    def transform_kwargs(self) -> Dict[str, Any]:
        '''
        No additional keyword arguments are required for `functional.to_image_and_mask`.
        Returns an empty dictionary.
        '''
        return {}


class SegRandomAffine(SegTransformBase):
    '''
    Random affine transform for both images and optional segmentation masks.
    For a single sample with an image and mask, the same random affine parameters are applied to both.

    This uses `functional.seg_random_affine` to apply transforms.

    Note: This supports separate fill values for the image and mask.
          The interpolation method for the image is also user-defined,
          while the method for the mask is always `InterpolationMode.NEAREST`.

    Args:
        degrees (Union[float, Sequence[float]]): 
            Range of degrees for rotational transform.
            If `Sequence[float]`, should represent `(min, max)`.
            If `float`, will assume `(-degrees, +degrees`).
            Default is 0.0 for no rotation.
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
            If `Sequence[float]`, should represent `(min_x, max_x)`
            for only x-axis shearing
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
    '''
    def __init__(
        self,
        degrees: Union[float, Sequence[float]] = 0.0,
        translate: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None, 
        shear: Optional[Union[int, float, Sequence[float]]] = None, 
        img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        img_fill: FillValue = 0,
        mask_fill: FillValue = 255
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


class SegRandomPerspective(SegTransformBase):
    '''
    Random perspective transform for both images and optional segmentation masks.
    For a single sample with an image and mask, the same random perspective parameters are applied to both.

    This uses `functional.seg_random_perspective` to apply transforms.

    Note: This supports separate fill values for the image and mask.
          The interpolation method for the image is also user-defined,
          while the method for the mask is always `InterpolationMode.NEAREST`.

    Args:
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
    '''
    def __init__(
        self,
        distortion_scale: float = 0.5, 
        p: float = 0.5,
        img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        img_fill: FillValue = 0,
        mask_fill: FillValue = 255
    ):
        self.distortion_scale = distortion_scale
        self.p = p
        self.img_interpolation = img_interpolation
        self.img_fill = img_fill
        self.mask_fill = mask_fill
        super().__init__(seg_transform = functional.seg_random_perspective)

    def __repr__(self) -> str:
        '''
        Returns a representational string for the `SegRandomPerspective` instance.
        '''
        repr_str = (
            f'SegRandomPerspective(distortion_scale = {self.distortion_scale}, p = {self.p}, '
            f'img_interpolation = {self.img_interpolation}, img_fill = {self.img_fill}, '
            f'mask_fill = {self.mask_fill})'
        )
        return repr_str
    
    @property
    def transform_kwargs(self) -> Dict[str, Any]:
        '''
        Returns the transform keyword arguments used for `functional.seg_random_perspective`.
        '''
        return {
            'distortion_scale': self.distortion_scale,
            'p': self.p,
            'img_interpolation': self.img_interpolation,
            'img_fill': self.img_fill,
            'mask_fill': self.mask_fill
        }


class SegLetterbox(SegTransformBase):
    '''
    Letterbox transform for both images and optional segmentation masks.
    This is similar to a standard resize transform, but the image is resized to fit 
    within the target dimensions while preserving the aspect ratio. 
    Any remaining space is filled with padding to match the target dimensions.

    This uses `functional.seg_letterbox` to apply transforms.

    Note: This supports separate fill values for the image and mask.
          The interpolation method for the image is also user-defined,
          while the method for the mask is always `InterpolationMode.NEAREST`.

    Args:
        size (SpatialSize): 
            Size `(height, width)` to transform the image and optional mask into,
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
    '''
    def __init__(
        self,
        size: SpatialSize, 
        img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR,
        img_fill: FillValue = 0,
        mask_fill: FillValue = 255
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
    
class SegResize(SegTransformBase):
    '''
    Resize transform for both images and optional segmentation masks.
    This will **not** preserve aspect ratio when resizing.
    
    This uses `functional.seg_resize` to apply transforms.

    Note: The interpolation method for the image is user-defined, 
          while the method for the mask is always `InterpolationMode.NEAREST`.

    Args:
        size (SpatialSize): 
            Size `(height, width)` to transform `img` and (optionally) `mask`.
            If `int`, assumed square.
        img_interpolation (Union[InterpolationMode, int]): 
            Interpolation mode used for the image transform.
            Default is `InterpolationMode.BILINEAR`.
    '''
    def __init__(
        self, 
        size: SpatialSize, 
        img_interpolation: Union[InterpolationMode, int] = InterpolationMode.BILINEAR
    ):
        super().__init__(seg_transform = functional.seg_resize)
        self.size = size
        self.img_interpolation = img_interpolation

    @property
    def transform_kwargs(self) -> Dict[str, Any]:
        '''
        Returns the transform keyword arguments used for `functional.seg_resize`.
        '''
        return {
            'size': self.size,
            'img_interpolation': self.img_interpolation
        }
