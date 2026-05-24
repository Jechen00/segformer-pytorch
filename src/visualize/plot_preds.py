#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torchvision.transforms import v2

import cv2
from matplotlib.figure import Figure
from typing import List, Optional, Union, Tuple, Literal, Dict

from src.inference import preprocess_and_predict
from src.visualize.format_plot_inputs import format_cls, format_seg
from src.visualize.figures import (
    make_cls_figure, make_seg_figure_collage, make_seg_figure_overlay
)
from src.data_setup.transforms.functional import reverse_letterbox_numpy
from src.utils import inverse_mapping

from src.ml_types import ImageInput, ImageLabel, RGBTuple


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
    figsize: Optional[Tuple[float, float]] = None,
    title_fontsize: float = 18
) -> Figure:
    '''
    Predicts image classification labels and display them as titles for the input images.
    If target labels are provided, the titles are 
    colored green for correct predictions and red for incorrect predictions.

    If multiple input images are provided, the results are laid out in a grid of subplots.
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


def plot_seg_preds_collage(
    model: nn.Module,
    imgs: Union[ImageInput, List[ImageInput]],
    idx_to_rgb: Dict[int, RGBTuple],
    transforms: Optional[v2.Compose] = None,
    unletterbox_preds: bool = False,
    targ_masks: Optional[Union[ImageInput, List[ImageInput]]] = None,
    targ_mode: Optional[Literal['index', 'rgb']] = None,
    fill_rgb: RGBTuple = (114, 114, 114),
    visible_classes: Optional[List[str]] = None,
    include_legend: bool = False,
    rgb_to_class: Optional[Dict[RGBTuple, str]] = None,
    memory_format: Optional[torch.memory_format] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title_fontsize: float = 18
):
    '''
    Predict segmentation index masks and creates a collage showing the
    input images, optional target masks, and prediction masks side-by-side.

    If multiple input images are provided, each result is shown on a separate row of the figure.
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

    # Reverse letterbox on prediction masks if needed
    if unletterbox_preds:
        for i, (img, pred) in enumerate(zip(imgs, pred_masks)):
            pred_masks[i] = reverse_letterbox_numpy(
                pred,
                orig_size = img.shape[:2],
                resize_to_orig = True,
                interpolation = cv2.INTER_NEAREST_EXACT,
                input_format = 'HWC'
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
    unletterbox_preds: bool = False,
    fill_rgb: RGBTuple = (114, 114, 114),
    visible_classes: Optional[List[str]] = None,
    include_legend: bool = False,
    rgb_to_class: Optional[Dict[RGBTuple, str]] = None,
    memory_format: Optional[torch.memory_format] = None,
    pred_alpha: float = 0.5,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None
):
    '''
    Predict segmentation index masks and plot them as overlays on top of the input images.
    
    If multiple input images are provided, the results are laid out in a grid of subplots.
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

    # Reverse letterbox on prediction masks if needed
    if unletterbox_preds:
        for i, (img, pred) in enumerate(zip(imgs, pred_masks)):
            pred_masks[i] = reverse_letterbox_numpy(
                pred,
                orig_size = img.shape[:2],
                resize_to_orig = True,
                interpolation = cv2.INTER_NEAREST_EXACT,
                input_format = 'HWC'
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