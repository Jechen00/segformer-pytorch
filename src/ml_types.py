#####################################
# Imports & Dependencies
#####################################
import torch
import numpy as np

from numbers import Real
from typing import (
    Union, Tuple, Dict, Sequence, 
    Literal, Optional, TypeAlias
)
from PIL import Image


#####################################
# Common Types
#####################################
RGBLike: TypeAlias = Union[int, Tuple[int, int, int]]

SpatialSize: TypeAlias = Union[int, Tuple[int, int]]
ImageInput: TypeAlias = Union[Image.Image, torch.Tensor]

MeasureValue: TypeAlias = Union[Real, np.ndarray, torch.Tensor]
MetricResult: TypeAlias = Union[MeasureValue, Dict[str, MeasureValue]]

MetricLogFields: TypeAlias = Sequence[Union[str, Tuple[str, Literal['mean', 'max', 'min']]]]
EntryLogUnits: TypeAlias = Optional[Union[str, Sequence[Optional[str]]]]