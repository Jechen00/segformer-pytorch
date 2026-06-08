#####################################
# Imports & Dependencies
#####################################
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.patches import Patch

import math
import numpy as np
from typing import List, Optional, Union, Tuple, Dict

from src.ml_types import PythonNum, RGBTuple
from src.utils.data_utils import make_tuple
from src.visualize.format_plot_inputs import NPImageList, IntLabelList

#####################################
# Image Classification Functions
#####################################
def make_cls_figure(
    imgs: NPImageList,
    pred_labels: IntLabelList,
    class_names: List[str],
    targ_labels: Optional[IntLabelList] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[PythonNum, PythonNum]] = None,
    title_fontsize: PythonNum = 18
) -> Figure: 
    '''
    Makes a figure that titles input images using their prediction labels.
    If target labels are provided, the titles are 
    colored green for correct predictions and red for incorrect predictions.

    If multiple input images (and their labels) are provided,
    they are laid out in a grid of subplots.

    Note: It is assumed that all labels are integers 
          in the range `[0, len(class_names)-1]`.

    Args:
        imgs (NPImageList): 
            Images to plot. This must be a list of ndarrays in HWC format.
        pred_labels (IntLabelList): 
            Prediction labels for `imgs`. This must be a list of integer indices.
        class_names (List[str]): 
            List of class names.
        targ_labels (optional, IntLabelList): 
            Target labels for `imgs`. This must be a list of integer indices.
        nrows (optional, int):
            Nunber of rows in the figure.
            If not provided, it is set to `math.ceil(num_imgs / ncols)`.
            This applies even when `ncols` is not provided, since `ncols` defaults to `math.ceil(math.sqrt(num_imgs))`.
            Here, `num_imgs = len(imgs)`.
        ncols (optional, int): 
            Number of columns in the figure.
            If not provided, but `nrows` is provided, it is set to `math.ceil(num_imgs / nrows)`.
            If not provided and `nrows` is not provided, it is set to `math.ceil(math.sqrt(num_imgs))`.
            Here, `num_imgs = len(imgs)`.
        figsize (optional, Tuple[PythonNum, PythonNum]): 
            Figure size in the form of a tuple (width, height).
            If not provided, defaults to `5 * ncols, 5 * nrows`.
        title_fontsize (PythonNum): 
            Fontsize for the titles of each subplot/panel. Default is `18`.

    Returns:
        Figure: 
            Matplotlib figure displaying the input images 
            along with their prediction labels and optional target labels.
    '''
    # Validate lengths
    num_imgs = len(imgs)
    if len(pred_labels) != num_imgs:
        raise ValueError(
            'Number of prediction labels (pred_labels) must match number of images (imgs).'
        )
    
    if targ_labels is None:
        targ_labels = [None] * num_imgs
    elif len(targ_labels) != num_imgs:
        raise ValueError(
            'Number of target labels (targ_labels) must match number of images (imgs).'
        )

    # Setup figure
    fig, axes = make_grid(num_imgs, nrows, ncols, figsize)
    flat_axes = axes.flatten() if num_imgs > 1 else [axes]

    # Plotting
    for ax, img, targ, pred in zip(flat_axes, imgs, targ_labels, pred_labels):
        # Plot image
        ax.imshow(img)

        # Make title
        title = f'Pred: {class_names[pred]}'
        clr = 'k'
        if targ is not None:
            title += f'\nTarg: {class_names[targ]}'
            clr = 'g' if (pred == targ) else 'r'

        ax.set_title(title, color = clr, fontsize = title_fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in flat_axes[num_imgs:]:
        ax.axis('off')

    fig.tight_layout(h_pad = 1.8)
    return fig


#####################################
# Semantic Segmentation Functions
#####################################
def make_seg_figure_collage(
    imgs: NPImageList,
    pred_masks: NPImageList,
    targ_masks: Optional[NPImageList] = None,
    include_legend: bool = False,
    rgb_to_class: Optional[Dict[RGBTuple, str]] = None,
    figsize: Optional[Tuple[PythonNum, PythonNum]] = None,
    title_fontsize: PythonNum = 18
) -> Figure:  
    '''
    Makes a figure that plots input images, optional target masks, and prediction masks
    side-by-side (creating a collage).

    If multiple input images (and their masks) are provided,
    each input is shown on a separate row of the figure.

    Args:
        imgs (NPImageList): 
            Images to plot. This must be a list of ndarrays in HWC format.
        pred_masks (NPImageList): 
            Prediction RGB masks to plot. This must be a list of ndarrays in HWC format.
        targ_masks (optional, NPImageList): 
            Target RGB masks to plot.
            This must be a list of ndarrays in HWC format.
            If provided, the figure will have 3 columns.
            Otherwise, it will have 2 columns (only `imgs` and `pred_masks`).
        include_legend (bool): 
            Whether to include a legend that maps the unique colors 
            in `pred_masks` and `targ_masks` to their class names.
            Default is `False`.
        rgb_to_class (optional, Dict[RGBTuple, str]): 
            Dictionary mapping RGB tuples to class names.
            This is used to create the legend and is required if `include_legend=True`.
            Ideally, all unique colors in `pred_masks` and `targ_masks` should be in this mapping.
            Missing entries are labeled as `Unknown`.
        figsize (optional, Tuple[PythonNum, PythonNum]): 
            Figure size in the form of a tuple (width, height).
            If not provided, defaults to `5 * ncols, 5 * len(imgs)`, where `ncols` is `2` or `3`.
        title_fontsize (PythonNum): 
            Fontsize for the titles of each subplot.panel. Default is `18`.

    Returns:
        Figure: 
            Matplotlib figure displaying collages of 
            the input images, optional target masks, and prediction masks.
    '''
    # Validate lengths
    num_imgs = len(imgs)
    if len(pred_masks) != num_imgs:
        raise ValueError(
            'Number of prediction masks (pred_masks) must match number of images (imgs).'
        )
    
    if targ_masks is None:
        targ_masks = [None] * num_imgs
    elif len(targ_masks) != num_imgs:
        raise ValueError(
            'Number of target masks (targ_masks) must match number of images (imgs).'
        )

    # Set up unique RGB set if legend is needed
    if include_legend:
        if rgb_to_class is None:
            raise ValueError(
                'If include_legend is True, '
                'RGB-to-class mapping (rgb_to_class) must be provided. '
            )
        unique_mask_rgbs = set()

    # Setup figure
    ncols = 2 if targ_masks[0] is None else 3
    pred_subplot_idx = ncols - 1 # Predictions are plotted on last column

    fig, axes = make_grid(num_imgs, num_imgs, ncols, figsize)
    axes = axes if num_imgs > 1 else [axes]

    # Plotting
    for row_ax, img, targ, pred in zip(axes, imgs, targ_masks, pred_masks):
        # Plot image
        row_ax[0].imshow(img)
        row_ax[0].set_title('Image', fontsize = title_fontsize)

        # Plot target mask (if provided)
        if targ is not None:
            row_ax[1].imshow(targ)
            row_ax[1].set_title('Target', fontsize = title_fontsize)

        # Plot prediction masks
        row_ax[pred_subplot_idx].imshow(pred)
        row_ax[pred_subplot_idx].set_title('Prediction', fontsize = title_fontsize)

        # Get unique colors between RGB masks
        if include_legend:
            pred_rgbs = pred[..., :3] # Shape: (height, width, 3)
            if pred.shape[-1] == 4:
                pred_rgbs = pred_rgbs[pred[..., 3] > 0] # Shape: (num_visible_pixels, 3)
            else:
                pred_rgbs = pred_rgbs.reshape(-1, 3) # Shape: (height * width, 3)
            unique_mask_rgbs.update(map(tuple, np.unique(pred_rgbs, axis = 0)))

            if targ is not None:
                targ_rgbs = targ[..., :3] # Shape: (height, width, 3)
                if targ.shape[-1] == 4:
                    targ_rgbs = targ_rgbs[targ[..., 3] > 0] # Shape: (num_visible_pixels, 3)
                else:
                    targ_rgbs = targ_rgbs.reshape(-1, 3) # Shape: (height * width, 3)
                unique_mask_rgbs.update(map(tuple, np.unique(targ_rgbs, axis = 0)))

        # Misc
        for i in range(ncols):
            row_ax[i].set_xticks([])
            row_ax[i].set_yticks([])

    # Create legend if needed
    if include_legend:
        handles = [
            Patch(facecolor = np.array(rgb) / 255,
                  label = rgb_to_class.get(rgb, 'Unknown').capitalize())
            for rgb in unique_mask_rgbs
        ]
        fig.legend(
            handles = handles,
            loc = 'center left',
            bbox_to_anchor = (1, 0.5),
            facecolor = 'lightgray',
            framealpha = 0.8
        )
        fig.subplots_adjust(right = 0.85)

    fig.tight_layout(h_pad = 3, w_pad = 0.1)
    return fig


def make_seg_figure_overlay(
    imgs: NPImageList,
    pred_masks: NPImageList,
    include_legend: bool = False,
    rgb_to_class: Optional[Dict[RGBTuple, str]] = None,
    pred_alpha: PythonNum = 0.5,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[PythonNum, PythonNum]] = None
) -> Figure:
    '''
    Makes a figure that overlays prediction segmentation masks on top of input images.

    If multiple input images (and their masks) are provided,
    they are laid out in a grid of subplots.

    Args:
        imgs (NPImageList): 
            Images to plot. This must be a list of ndarrays in HWC format.
        pred_masks (NPImageList): 
            Prediction RGB masks to plot. This must be a list of ndarrays in HWC format.
        include_legend (bool): 
            Whether to include a legend that maps the unique colors 
            in `pred_masks` and `targ_masks` to their class names.
            Default is `False`.
        rgb_to_class (optional, Dict[RGBTuple, str]): 
            Dictionary mapping RGB tuples to class names.
            This is used to create the legend and is required if `include_legend=True`.
            Ideally, all unique colors in `pred_masks` should be in this mapping.
            Missing entries are labeled as `Unknown`.
        pred_alpha (PythonNum): 
            The opacity of the prediction RGB masks when overlaid on top of the images.
            Must be in the range [0, 1], where `0` is fully transparent and `1` is fully opaque.
            Default is `0.5`.
        nrows (optional, int): 
            Nunber of rows in the figure.
            If not provided, it is set to `math.ceil(num_imgs / ncols)`.
            This applies even when `ncols` is not provided, since `ncols` defaults to `math.ceil(math.sqrt(num_imgs))`.
            Here, `num_imgs = len(imgs)`.
        ncols (optional, int): 
            Number of columns in the figure.
            If not provided, but `nrows` is provided, it is set to `math.ceil(num_imgs / nrows)`.
            If not provided and `nrows` is not provided, it is set to `math.ceil(math.sqrt(num_imgs))`.
            Here, `num_imgs = len(imgs)`.
        figsize (optional, Tuple[PythonNum, PythonNum]): 
            Figure size in the form of a tuple (width, height).
            If not provided, defaults to `5 * ncols, 5 * nrows`.

    Returns:
        Figure: 
            Matplotlib figure displaying the input images with overlaid prediction masks.
    '''
    # Validate lengths
    num_imgs = len(imgs)
    if len(pred_masks) != num_imgs:
        raise ValueError(
            'Number of prediction masks (pred_masks) must match number of images (imgs).'
        )

    # Set up unique RGB set if legend is needed
    if include_legend:
        if rgb_to_class is None:
            raise ValueError(
                'If include_legend is True, '
                'RGB-to-class mapping (rgb_to_class) must be provided. '
            )
        unique_mask_rgbs = set()

    # Setup figure
    fig, axes = make_grid(num_imgs, nrows, ncols, figsize)
    flat_axes = axes.flatten() if num_imgs > 1 else [axes]

    # Plotting
    for ax, img, pred in zip(flat_axes, imgs, pred_masks):
        # Plot image
        ax.imshow(img)

         # Plot prediction mask
        ax.imshow(pred, alpha = pred_alpha)

        # Get unique colors between RGB masks
        if include_legend:
            pred_rgbs = pred[..., :3] # Shape: (height, width, 3)
            if pred.shape[-1] == 4:
                pred_rgbs = pred_rgbs[pred[..., 3] > 0] # Shape: (num_visible_pixels, 3)
            else:
                pred_rgbs = pred_rgbs.reshape(-1, 3) # Shape: (height * width, 3)
            unique_mask_rgbs.update(map(tuple, np.unique(pred_rgbs, axis = 0)))

        ax.set_xticks([])
        ax.set_yticks([])

    # Misc
    for ax in flat_axes[num_imgs:]:
        ax.axis('off')

    # Create legend if needed
    if include_legend:
        handles = [
            Patch(facecolor = np.array(rgb) / 255,
                  label = rgb_to_class.get(rgb, 'Unknown').capitalize())
            for rgb in unique_mask_rgbs
        ]
        fig.legend(
            handles = handles,
            loc = 'center left',
            bbox_to_anchor = (1, 0.5),
            facecolor = 'lightgray',
            framealpha = 0.8
        )
        fig.subplots_adjust(right = 0.85)

    for ax in flat_axes[num_imgs:]:
        ax.axis('off')

    fig.tight_layout()
    return fig


#####################################
# Figure Utility Functions
#####################################
def make_grid(
    min_panels: int,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[PythonNum, PythonNum]] = None,
    panel_scale: Union[PythonNum, Tuple[PythonNum, PythonNum]] = 5.0
) -> Tuple[Figure, Union[Axes, np.ndarray]]:
    '''
    Creates a figure with a grid of subplots containing at least `min_panels` panels.

    Note: If `nrows` and `ncols` are both provided, must have `(nrows * ncols) < min_panels`.

    Args:
        min_panels (int): 
            Minimum number of panels on the grid.
        nrows (optional, int): 
            Nunber of rows in the grid.
            If not provided, it is set to `math.ceil(min_panels / ncols)`.
            This applies even when `ncols` is not provided, since `ncols` defaults to `math.ceil(math.sqrt(min_panels))`.
        ncols (optional, int): 
            Number of columns in the grid.
            If not provided, but `nrows` is provided, it is set to `math.ceil(min_panels / nrows)`.
            If not provided and `nrows` is not provided, it is set to `math.ceil(math.sqrt(min_panels))`.
        figsize (optional, Tuple[PythonNum, PythonNum]): 
            Figure size in the form of a tuple (width, height).
            If not provided, defaults to `panel_scale[0] * ncols, panel_scale[1] * nrows`.
        panel_scale (Union[PythonNum, Tuple[PythonNum, PythonNum]]): 
            The scale of each panel in the form of a tuple (width, height).
            This is used to determine the figure size if `figsize` is not provided.
            If provided as a single number, it is assumed square.
            Default is `5.0`.

    Returns:
        fig (Figure): 
            The matplotlib figure with the grid of subplots.
        axes (Union[Axes, np.ndarray]):
            The matplotlib Axes objects for `fig`.
            If `nrows=1` and `ncols=1`, this is a single Axes object.
            Otherwise, this is a ndarray of Axes objects aranged in a grid.
    '''
    panel_scale = make_tuple(panel_scale)
    
    if (nrows is not None) and (ncols is not None):
        if (nrows * ncols) < min_panels:
            raise ValueError(
                'If both nrows and ncols are provided, must have (nrows * ncols) >= min_panels. '
                f'Got: {nrows * ncols} < {min_panels} .'
            )
    elif ncols is not None:
        nrows = math.ceil(min_panels / ncols)
    elif nrows is not None:
        ncols = math.ceil(min_panels / nrows)
    else:
        ncols = math.ceil(math.sqrt(min_panels))
        nrows = math.ceil(min_panels / ncols)

    figsize = (panel_scale[0] * ncols, panel_scale[1] * nrows) if figsize is None else figsize

    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = figsize)

    plt.close(fig)
    return fig, axes