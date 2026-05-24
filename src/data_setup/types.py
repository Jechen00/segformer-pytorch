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
SegSample: TypeAlias = SegDict[ImageInput, ImageInput]

SampleDict: TypeAlias = Union[ImageDict[ImageInput], ClsSample, SegSample]
Sample: TypeAlias = Union[ImageInput, SampleDict]


#####################################
# Typles for Multiple Samples
#####################################
ClsSampleList: TypeAlias = ClsDict[List[ImageInput], List[ImageLabel]]
SegSampleList: TypeAlias = SegDict[List[ImageInput], List[ImageInput]]

SampleListDict: TypeAlias = Union[
    ImageDict[List[ImageInput]],
    ClsSampleList,
    SegSampleList
]

# torch.Tensor represents a collated batch of tensors
MultiSamples: TypeAlias = Union[torch.Tensor, List[Sample], SampleListDict]