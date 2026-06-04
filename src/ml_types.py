#####################################
# Imports & Dependencies
#####################################
import torch
import numpy as np

from PIL import Image
from typing import Union, List, Tuple, Literal, Sequence, TypeAlias


#####################################
# Types
#####################################
PythonNum: TypeAlias = Union[int, float]

ImageInput: TypeAlias = Union[Image.Image, torch.Tensor]
ImageLabel: TypeAlias = Union[int, torch.Tensor]

RGBTuple: TypeAlias = Tuple[int, int, int]
FillValue: TypeAlias = Union[float, int, Sequence[float], Sequence[int]]

SpatialSize: TypeAlias = Union[int, Tuple[int, int]]

IndexLike: TypeAlias = Union[int, List[int], torch.Tensor, np.ndarray] # 1D indexing

Aggregation: TypeAlias = Literal['mean', 'median', 'max', 'min']