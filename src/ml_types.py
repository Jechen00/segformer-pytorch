#####################################
# Imports & Dependencies
#####################################
import torch
import numpy as np

from PIL import Image
from typing import Union, List, Tuple, Literal, TypeAlias


#####################################
# Types
#####################################
ImageInput: TypeAlias = Union[Image.Image, torch.Tensor]
ImageLabel: TypeAlias = Union[int, torch.Tensor]

RGBTuple: TypeAlias = Tuple[int, int, int]
RGBLike: TypeAlias = Union[int, RGBTuple]

SpatialSize: TypeAlias = Union[int, Tuple[int, int]]

IndexLike: TypeAlias = Union[int, List[int], torch.Tensor, np.ndarray] # 1D indexing

Aggregation: TypeAlias = Literal['mean', 'median', 'max', 'min']