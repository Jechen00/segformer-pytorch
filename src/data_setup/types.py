#####################################
# Imports & Dependencies
#####################################
import torch

from typing import (
    Union, List, TypeAlias, 
    Generic, TypeVar, TypedDict
)

from src.ml_types import ImageInput, ImageLabel

from typing import Generic, TypeVar


#####################################
# TypeVars and Generics
#####################################
I = TypeVar('I')
L = TypeVar('L')
M = TypeVar('M')

class ImageDict(TypedDict, Generic[I]):
    image: I

class ClsDict(TypedDict, Generic[I, L]):
    image: I
    label: L

class SegDict(TypedDict, Generic[I, M]):
    image: I
    mask: M


#####################################
# Types for Single Sample
#####################################
ClsSample: TypeAlias = ClsDict[ImageInput, ImageLabel]
'''
Dictionary containing image classification information for a single sample.
This should have the fields (non-exhaustive):
    - image (ImageInput): Image sample, as a PIL image or tensor.
    - label (ImageLabel): Class label, as an integer or tensor.
'''

SegSample: TypeAlias = SegDict[ImageInput, ImageInput]
'''
Dictionary containing segmentation information for a single sample.
This should have the fields (non-exhaustive):
    - image (ImageInput): Image sample, as a PIL image or tensor.
    - mask (ImageInput): Segmentation mask, as a PIL image or tensor.
'''

SampleDict: TypeAlias = Union[ImageDict[ImageInput], ClsSample, SegSample]
Sample: TypeAlias = Union[ImageInput, SampleDict]


#####################################
# Typles for Multiple Samples
#####################################
ClsSampleList: TypeAlias = ClsDict[List[ImageInput], List[ImageLabel]]
'''
Dictionary containing image classification information for multiple samples.
This should have the fields (non-exhaustive):
    - image (List[ImageInput]): List of image samples. Each sample is a PIL image or tensor.
    - label (List[ImageLabel]): List of class labels for the images in `image`.
                                Each label is an integer or tensor.
'''

SegSampleList: TypeAlias = SegDict[List[ImageInput], List[ImageInput]]
'''
Dictionary containing segmentation information for multiple samples.
This should have the fields (non-exhaustive):
    - image (List[ImageInput]): List of image samples. Each sample is a PIL image or tensor.
    - mask (List[ImageInput]): List of segmentation masks for the images in `image`.
                               Each mask is a PIL image or tensor.
'''

SampleListDict: TypeAlias = Union[
    ImageDict[List[ImageInput]],
    ClsSampleList,
    SegSampleList
]

# torch.Tensor represents a collated batch of tensors
MultiSamples: TypeAlias = Union[torch.Tensor, List[Sample], SampleListDict]