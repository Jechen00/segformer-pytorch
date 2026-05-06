#####################################
# Imports & Dependencies
#####################################
import matplotlib.pyplot as plt
import math

from typing import Optional, Tuple, Union
from src.utils import make_tuple


#####################################
# Functions
#####################################
def make_grid(
    min_panels: int,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    panel_scale: Union[float, Tuple[float, float]] = 5.0
):
    '''
    min_panels (int): Minimum number of panels on the grid.
    '''
    panel_scale = make_tuple(panel_scale)
    
    if (nrows is not None) and (ncols is not None):
        if (nrows * ncols) < min_panels:
            raise ValueError(
                'If both nrows and ncols are provided, must have (nrows * ncols) >= min_panels. '
                f'Got: {nrows * ncols} < {min_panels} .'
            )
    elif nrows is not None:
        ncols = math.ceil(min_panels / nrows)
    elif ncols is not None:
        nrows = math.ceil(min_panels / ncols)
    else:
        nrows = math.ceil(math.sqrt(min_panels))
        ncols = math.ceil(min_panels / nrows)

    figsize = (panel_scale[0] * ncols, panel_scale[1] * nrows) if figsize is None else figsize

    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)

    plt.close(fig)
    return fig, axes