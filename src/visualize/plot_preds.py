#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torchvision.transforms import v2

import cv2
import numpy as np
from matplotlib.figure import Figure

from typing import List, Optional, Union, Tuple, Literal, Dict

from src.visualize.format_plot_inputs import format_cls, format_seg
from src.visualize.figures import (
    make_cls_figure, make_seg_figure_collage, make_seg_figure_overlay
)

from src.inference import preprocess_and_predict
from src.data_setup.transforms.functional import reverse_letterbox_numpy

from src.utils.data_utils import inverse_mapping
from src.ml_types import PythonNum, ImageInput, ImageLabel, RGBTuple


#####################################
# Image Classification Functions
#####################################
def plot_cls_preds(
    model: nn.Module,
    imgs: Union[ImageInput, List[ImageInput]],
    class_names: List[str],
    targ_labels: Union[ImageLabel, List[ImageLabel]] = None,
    transforms: Optional[v2.Compose] = None,
    memory_format: Optional[torch.memory_format] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[PythonNum, PythonNum]] = None,
    title_fontsize: PythonNum = 18
) -> Figure:
    '''
    Predicts image classification labels and display them as titles for the input images.
    If target labels are provided, the titles are 
    colored green for correct predictions and red for incorrect predictions.

    If multiple input images are provided, the results are laid out in a grid of subplots.

    Args:
        model (nn.Module):
            Image classification model used to make prediction labels for `imgs`.
        imgs (Union[ImageInput, List[ImageInput]]): 
            Images to prediction on and plot.
            This supports:
                - A single PIL image
                - A single 3D tensor of shape (channels, height, width)
                - A list of images (each element is a PIL image or 3D tensor)
                - A batched 4D tensor of shape (batch_size, channels, height, width)
        class_names (List[str]): 
            List of class names.
        targ_labels (Union[ImageLabel, List[ImageLabel]]): 
            Target labels for `imgs`.
            This supports:
                - A single integer
                - A single-element tensor
                - A list of labels (each element is an integer or single-element tensor)
                - A batched 1D tensor of shape (batch_size,)
        transforms (optional, v2.Compose): 
            Transforms used to preprocess `imgs` before passing them into `model`.
            If provided, ensure that the `imgs` are transformed to a tensor by the end of the pipeline.
            If not provided, ensure that `imgs` are already tensors.
        memory_format (optional, torch.memory_format): 
            The memory format to convert `imgs` to before predicting.
            This should ideally be the same memory format as `model`,
            but it is not required.
            If not provided, no memory format conversion is applied.
        nrows (optional, int): 
            Nunber of rows in the figure.
            If not provided, it is set to `math.ceil(num_imgs / ncols)`.
            This applies even when `ncols` is not provided, since `ncols` defaults to `math.ceil(math.sqrt(num_imgs))`.
            Here, `num_imgs=len(imgs)` after internally converting `imgs` to a list.
        ncols (optional, int): 
            Number of columns in the figure.
            If not provided, but `nrows` is provided, it is set to `math.ceil(num_imgs / nrows)`.
            If not provided and `nrows` is not provided, it is set to `math.ceil(math.sqrt(num_imgs))`.
            Here, `num_imgs=len(imgs)` after internally converting `imgs` to a list.
        figsize (optional, Tuple[PythonNum, PythonNum]): 
            Figure size in the form of a tuple (width, height).
            If not provided, defaults to `5 * ncols, 5 * nrows`.
        title_fontsize (PythonNum): 
            Fontsize for the titles of each subplot/panel. Default is `18`.

    Returns:
        Figure: 
            Matplotlib figure displaying the input images along with their prediction labels
            and optional target labels.
    '''
    # Predictions
    pred_labels = preprocess_and_predict(model, imgs, transforms, memory_format)

    # Formatting
    imgs, pred_labels, targ_labels = format_cls(imgs, pred_labels, targ_labels)

    # Plotting
    return make_cls_figure(
        imgs = imgs,
        pred_labels = pred_labels,
        class_names = class_names,
        targ_labels = targ_labels,
        nrows = nrows,
        ncols = ncols,
        figsize = figsize,
        title_fontsize = title_fontsize
    )


#####################################
# Segmentation Functions
#####################################
def plot_seg_preds_collage(
    model: nn.Module,
    imgs: Union[ImageInput, List[ImageInput]],
    idx_to_rgb: Dict[int, RGBTuple],
    transforms: Optional[v2.Compose] = None,
    pred_rev_sizing_mode: Optional[Literal['resize', 'letterbox']] = None,
    targ_masks: Optional[Union[ImageInput, List[ImageInput]]] = None,
    targ_mode: Optional[Literal['index', 'rgb']] = None,
    fill_rgb: RGBTuple = (114, 114, 114),
    visible_classes: Optional[List[str]] = None,
    include_legend: bool = False,
    rgb_to_class: Optional[Dict[RGBTuple, str]] = None,
    memory_format: Optional[torch.memory_format] = None,
    figsize: Optional[Tuple[PythonNum, PythonNum]] = None,
    title_fontsize: PythonNum = 18
):
    '''
    Predict segmentation index masks and creates a collage showing the
    input images, optional target masks, and prediction masks side-by-side.

    If multiple input images are provided, each result is shown on a separate row of the figure.

    Args:
        model (nn.Module):
            Segmentation model used to make prediction masks for `imgs`.
        imgs (Union[ImageInput, List[ImageInput]]):
            Images to prediction on and plot.
            This supports:
                - A single PIL image
                - A single 3D tensor of shape (channels, height, width)
                - A list of images (each element is a PIL image or 3D tensor)
                - A batched 4D tensor of shape (batch_size, channels, height, width)
        idx_to_rgb (Dict[int, RGBTuple]):
            Dictionary mapping integer indices to RGB tuples.
            This mapping should be one-to-one (injective) and the RGB values should be in [0, 255].
        transforms (optional, v2.Compose):
            Transforms used to preprocess `imgs` before passing them into `model`.
            If provided, ensure that the `imgs` are transformed to a tensor by the end of the pipeline.
            If not provided, ensure that `imgs` are already tensors.
        pred_rev_sizing_mode (optional, Literal['resize', 'letterbox']):
            If provided, restores the prediction masks to their original image sizes
            by reversing the preprocessing sizing method (letterbox or resize).
                - 'letterbox': The letterbox padding is first removed from the prediction masks
                               before resizing them to their original image sizes.
                - 'resize': Prediction masks are directly resized to their original image sizes.
            If not provided (`None`), prediction masks are plotted as-is, with no resizing.
        targ_masks (optional, Union[ImageInput, List[ImageInput]]):
            Target masks for `imgs`. These may be index or RGB masks.
            This supports:
                - A single PIL image
                - A tensor of a single mask:
                    - RGB mode: shape must be (3, height, width)
                    - Index mode: shape must be (height, width)
                - A list of masks: Each element is a single mask (PIL image or tensor), following the rules above.
                - A batched tensor of multiple masks:
                    - RGB mode: shape must be (batch_size, 3, height, width)
                    - Index mode: shape must be (batch_size, height, width)
            This argument is required if `targ_mode` is provided.
        targ_mode (optional, Literal[rgb', 'index']): 
            The mode of the target masks.
                - 'rgb': Target masks are expected to be RGB masks.
                - 'index': Target masks are expected to be index masks.
            This argument is required if `targ_masks` is provided.
        fill_rgb (RGBTuple): 
            RGB tuple used to fill in pixels whose index is not present in `idx_to_rgb`.
            The RGB values should be in [0, 255]. Default is `(114, 114, 114)`.
        visible_classes (optional, List[str]):
            List of classes to set as visible in the prediction masks and `targ_masks`.
            If provided, classes not in this list will be fully transparent.
        include_legend (bool):
            Whether to include a legend that maps the unique colors 
            in the prediction masks and `targ_masks` to their class names.
            Default is `False`.
        rgb_to_class (optional, Dict[RGBTuple, str]):
            Dictionary mapping RGB tuples to class names.
            Required if `visible_classes` is provided or if `include_legend=True`.
                - If `visible_classes` is provided: 
                    All classes in `visible_classes` must have an entry in this mapping.
                - If `include_legend=True`: 
                    Ideally all unique colors in the prediction masks and `targ_masks` should be in this mapping. 
                    Missing entries will be labeled as `Unknown`.
        memory_format (optional, torch.memory_format): 
            The memory format to convert `imgs` to before predicting.
            This should ideally be the same memory format as `model`,
            but it is not required.
            If not provided, no memory format conversion is applied.
        figsize (optional, Tuple[PythonNum, PythonNum]): 
            Figure size in the form of a tuple (width, height).
            If not provided, defaults to `5 * ncols, 5 * len(imgs)`, where `ncols` is `2` or `3`. 
            Note that `len(imgs)` is computed after internally converting `imgs` to a list.
        title_fontsize (PythonNum): 
            Fontsize for the titles of each subplot/panel. Default is `18`.

    Returns:
        Figure:
            Matplotlib figure displaying collages of 
            the input images, optional target masks, and prediction masks.
    '''
    # Validate that required mappings are present
    if visible_classes is not None:
        if rgb_to_class is None:
            raise ValueError(
                'If visible_classes is provided, '
                'RGB-to-class mapping (rgb_to_class) must also be provided. '
            )
        try:
            class_to_rgb = inverse_mapping(rgb_to_class)
            visible_rgbs = [class_to_rgb[name] for name in visible_classes]
        except Exception as e:
            raise ValueError('Unable to create class-to-RGB mapping.') from e
    else:
        visible_rgbs = None

    # Predictions
    pred_masks = preprocess_and_predict(model, imgs, transforms, memory_format)
    
    # Formatting
    imgs, pred_masks, targ_masks = format_seg(
        imgs = imgs, 
        pred_masks = pred_masks, 
        targ_masks = targ_masks,
        idx_to_rgb = idx_to_rgb,
        targ_mode = targ_mode,
        fill_rgb = fill_rgb,
        visible_rgbs = visible_rgbs
    )

    # Reverse sizing on prediction masks if needed
    if pred_rev_sizing_mode is not None:
        for i, (img, pred) in enumerate(zip(imgs, pred_masks)):
            pred_masks[i] = reverse_pred_sizing(
                pred = pred, 
                orig_size = img.shape[:2], 
                mode = pred_rev_sizing_mode
            )

    if (rgb_to_class is not None) and (fill_rgb not in rgb_to_class):
        rgb_to_class[fill_rgb] = 'Filler'

    # Plotting
    return make_seg_figure_collage(
        imgs = imgs,
        pred_masks = pred_masks,
        targ_masks = targ_masks,
        include_legend = include_legend,
        rgb_to_class = rgb_to_class,
        figsize = figsize,
        title_fontsize = title_fontsize
    )


def plot_seg_preds_overlay(
    model: nn.Module,
    imgs: Union[ImageInput, List[ImageInput]],
    idx_to_rgb: Dict[int, RGBTuple],
    transforms: Optional[v2.Compose] = None,
    pred_rev_sizing_mode: Optional[Literal['resize', 'letterbox']] = None,
    fill_rgb: RGBTuple = (114, 114, 114),
    visible_classes: Optional[List[str]] = None,
    include_legend: bool = False,
    rgb_to_class: Optional[Dict[RGBTuple, str]] = None,
    memory_format: Optional[torch.memory_format] = None,
    pred_alpha: PythonNum = 0.5,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[PythonNum, PythonNum]] = None
):
    '''
    Predict segmentation index masks and plot them as overlays on top of the input images.
    
    If multiple input images are provided, the results are laid out in a grid of subplots.

    Args:
        model (nn.Module):
            Segmentation model used to make prediction masks for `imgs`.
        imgs (Union[ImageInput, List[ImageInput]]):
            Images to prediction on and plot.
            This supports:
                - A single PIL image
                - A single 3D tensor of shape (channels, height, width)
                - A list of images (each element is a PIL image or 3D tensor)
                - A batched 4D tensor of shape (batch_size, channels, height, width)
        idx_to_rgb (Dict[int, RGBTuple]):
            Dictionary mapping integer indices to RGB tuples.
            This mapping should be one-to-one (injective) and the RGB values should be in [0, 255].
        transforms (optional, v2.Compose):
            Transforms used to preprocess `imgs` before passing them into `model`.
            If provided, ensure that the `imgs` are transformed to a tensor by the end of the pipeline.
            If not provided, ensure that `imgs` are already tensors.
        pred_rev_sizing_mode (optional, Literal['resize', 'letterbox']):
            If provided, restores the prediction masks to their original image sizes
            by reversing the preprocessing sizing method (letterbox or resize).
                - 'letterbox': The letterbox padding is first removed from the prediction masks
                               before resizing them to their original image sizes.
                - 'resize': Prediction masks are directly resized to their original image sizes.
            If not provided (`None`), prediction masks are plotted as-is, with no resizing.
        fill_rgb (RGBTuple): 
            RGB tuple used to fill in pixels whose index is not present in `idx_to_rgb`.
            The RGB values should be in [0, 255]. Default is `(114, 114, 114)`.
        visible_classes (optional, List[str]):
            List of classes to set as visible in the prediction masks and `targ_masks`.
            If provided, classes not in this list will be fully transparent.
        include_legend (bool):
            Whether to include a legend that maps the unique colors 
            in the prediction masks to their class names.
            Default is `False`.
        rgb_to_class (optional, Dict[RGBTuple, str]):
            Dictionary mapping RGB tuples to class names.
            Required if `visible_classes` is provided or if `include_legend=True`.
                - If `visible_classes` is provided: 
                    All classes in `visible_classes` must have an entry in this mapping.
                - If `include_legend=True`: 
                    Ideally all unique colors in the prediction masks should be in this mapping. 
                    Missing entries will be labeled as `Unknown`.
        memory_format (optional, torch.memory_format): 
            The memory format to convert `imgs` to before predicting.
            This should ideally be the same memory format as `model`,
            but it is not required.
            If not provided, no memory format conversion is applied.
        nrows (optional, int): 
            Nunber of rows in the figure.
            If not provided, it is set to `math.ceil(num_imgs / ncols)`.
            This applies even when `ncols` is not provided, since `ncols` defaults to `math.ceil(math.sqrt(num_imgs))`.
            Here, `num_imgs=len(imgs)` after internally converting `imgs` to a list.
        ncols (optional, int): 
            Number of columns in the figure.
            If not provided, but `nrows` is provided, it is set to `math.ceil(num_imgs / nrows)`.
            If not provided and `nrows` is not provided, it is set to `math.ceil(math.sqrt(num_imgs))`.
            Here, `num_imgs=len(imgs)` after internally converting `imgs` to a list.
        figsize (optional, Tuple[PythonNum, PythonNum]): 
            Figure size in the form of a tuple (width, height).
            If not provided, defaults to `5 * ncols, 5 * len(imgs)`, where `ncols` is `2` or `3`. 
            Note that `len(imgs)` is computed after internally converting `imgs` to a list.
        title_fontsize (PythonNum): 
            Fontsize for the titles of each subplot/panel. Default is `18`.

    Returns:
        Figure:
            Matplotlib figure displaying the input images with overlaid prediction masks.
    '''
    # Validate that required mappings are present
    if visible_classes is not None:
        if rgb_to_class is None:
            raise ValueError(
                'If visible_classes is provided, '
                'RGB-to-class mapping (rgb_to_class) must also be provided. '
            )
        try:
            class_to_rgb = inverse_mapping(rgb_to_class)
            visible_rgbs = [class_to_rgb[name] for name in visible_classes]
        except Exception as e:
            raise ValueError('Unable to create class-to-RGB mapping.') from e
    else:
        visible_rgbs = None

    # Predictions
    pred_masks = preprocess_and_predict(model, imgs, transforms, memory_format)
    
    # Formatting
    imgs, pred_masks, _ = format_seg(
        imgs = imgs, 
        pred_masks = pred_masks, 
        idx_to_rgb = idx_to_rgb,
        fill_rgb = fill_rgb,
        visible_rgbs = visible_rgbs
    )

    # Reverse sizing on prediction masks if needed
    if pred_rev_sizing_mode is not None:
        for i, (img, pred) in enumerate(zip(imgs, pred_masks)):
            pred_masks[i] = reverse_pred_sizing(
                pred = pred, 
                orig_size = img.shape[:2], 
                mode = pred_rev_sizing_mode
            )

    if (rgb_to_class is not None) and (fill_rgb not in rgb_to_class):
        rgb_to_class[fill_rgb] = 'Filler'

    # Plotting
    return make_seg_figure_overlay(
        imgs = imgs,
        pred_masks = pred_masks,
        include_legend = include_legend,
        rgb_to_class = rgb_to_class,
        pred_alpha = pred_alpha,
        nrows = nrows,
        ncols = ncols,
        figsize = figsize
    )


# -----------------------------------
# Helper Functions
# -----------------------------------
def reverse_pred_sizing(
    pred: np.ndarray, 
    orig_size: Tuple[int, int], 
    mode: Literal['letterbox', 'resize']
) -> np.ndarray:
    '''
    Restores a prediction mask to the original image size 
    by reversing the preprocessing sizing method (letterbox or resize).

    Note: The interpolation method used is always `cv2.INTER_NEAREST_EXACT`.

    Args:
        pred (np.ndarray): 
            Prediction RGB mask in the form of a ndarray.
            This must be in HWC format.
        orig_size (Tuple[int, int]): 
            Original image size as a tuple (height, width).
        mode (Literal['letterbox', 'resize']): 
            The preprocessing sizing mode applied to the image
            before passing it into the model to make a prediction mask.
            If `mode='letterbox'`, the letterbox padding is first removed from `pred`
                                   before resizing it to `orig_size`.
            If `mode='resize'`, `pred` is directly resized to `orig_size`.

    Returns:
        np.ndarray: 
            The restored prediction mask with shape (orig_size[0], orig_size[1], channels)
    '''
    if mode == 'letterbox':
        return reverse_letterbox_numpy(
            pred,
            orig_size = orig_size,
            interpolation = cv2.INTER_NEAREST_EXACT
        )
    elif mode == 'resize':
        return cv2.resize(
            pred,
            dsize = orig_size[::-1], # (width, height)
            interpolation = cv2.INTER_NEAREST_EXACT
        )
    else:
        raise ValueError("mode must be 'letterbox' or 'resize'.")