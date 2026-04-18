#####################################
# Imports & Dependencies
#####################################
import torch
import numpy as np
from typing import Union, Tuple, Dict
from PIL import Image


#####################################
# Common Types
#####################################
SpatialSize = Union[int, Tuple[int, int]]
ImageInput = Union[Image.Image, torch.Tensor]

