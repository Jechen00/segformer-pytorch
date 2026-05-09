#####################################
# Imports & Dependencies
#####################################
from __future__ import annotations

import torch
import numpy as np

from numbers import Real
from typing import (
    Union, Tuple, Dict, Sequence, 
    Literal, Optional, TypeAlias,
    List, TypedDict, NotRequired
)
from PIL import Image


#####################################
# Common Types
#####################################
class SampleDict(TypedDict):
    image: ImageInput
    label: NotRequired[ImageLabel]
    mask: NotRequired[ImageInput]

class SampleListDict(TypedDict):
    image: List[ImageInput]
    label: NotRequired[List[ImageLabel]]
    mask: NotRequired[List[ImageInput]]

class CollatedDict(TypedDict):
    image: torch.Tensor # Shape is (batch_size, 3, height, width)
    label: NotRequired[torch.Tensor] # Shape is (batch_size,)
    mask: NotRequired[torch.Tensor] # Shape is (batch_size, height, width)

    
IndexLike: TypeAlias = Union[int, List[int], Tuple[int, ...], torch.Tensor, np.ndarray]
RGBTuple: TypeAlias = Tuple[int, int, int]
RGBLike: TypeAlias = Union[int, RGBTuple]

SpatialSize: TypeAlias = Union[int, Tuple[int, int]]
ImageInput: TypeAlias = Union[Image.Image, torch.Tensor]
ImageLabel: TypeAlias = Union[int, torch.Tensor]

MeasureValue: TypeAlias = Union[Real, np.ndarray, torch.Tensor]
MetricGroup: TypeAlias = Dict[str, MeasureValue]
MetricResults: TypeAlias = Dict[str, Union[MeasureValue, MetricGroup]]

Agg: TypeAlias = Literal['mean', 'max', 'min']
MetricLogFields: TypeAlias = Sequence[Union[str, Tuple[str, Agg]]]
EntryLogUnits: TypeAlias = Optional[Union[str, Sequence[Optional[str]]]]

Sample: TypeAlias = Union[ImageInput, SampleDict]
CollatedSamples: TypeAlias = Union[torch.Tensor, CollatedDict]
BatchedSamples: TypeAlias = Union[List[Sample], SampleListDict, CollatedSamples]