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
RGBLike: TypeAlias = Union[int, Tuple[int, int, int]]

SpatialSize: TypeAlias = Union[int, Tuple[int, int]]
ImageInput: TypeAlias = Union[Image.Image, torch.Tensor]
ImageLabel: TypeAlias = Union[int, torch.Tensor]

MeasureValue: TypeAlias = Union[Real, np.ndarray, torch.Tensor]
MetricGroup: TypeAlias = Dict[str, MeasureValue]
MetricResults: TypeAlias = Dict[str, Union[MeasureValue, MetricGroup]]

Agg: TypeAlias = Literal['mean', 'max', 'min']
MetricLogFields: TypeAlias = Sequence[Union[str, Tuple[str, Agg]]]
EntryLogUnits: TypeAlias = Optional[Union[str, Sequence[Optional[str]]]]

class SampleDict(TypedDict):
    image: ImageInput
    label: NotRequired[ImageLabel]
    mask: NotRequired[ImageInput]

class SampleListDict(TypedDict):
    image: List[ImageInput]
    label: NotRequired[List[ImageLabel]]
    mask: NotRequired[List[ImageInput]]