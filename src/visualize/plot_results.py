#####################################
# Imports & Dependencies
#####################################
import torch

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, List, Dict

from src.logging.history import TrainHistory, ValHistory
from src.visualize.figures import make_grid
from src.utils.data_utils import nested_extract
from src.metrics.postprocess import (
    MetricSpecLike, format_metric_spec, select_and_agg_scalar_metric
)
from src.ml_types import PythonNum


#####################################
# Functions
#####################################
def plot_loss(
    train_history: Optional[TrainHistory],
    val_history: Optional[ValHistory],
    agg_label: Optional[str] = 'Avg',
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    '''
    Plots the training and validation loss curves.

    Note: At least one of `train_history` and `val_history` must be provided.

    Args:
        train_history (optional, TrainHistory):
            `TrainHistory` instance containing the training loss curve.
            If not provided, no training loss curve is plotted.
        val_history (optional, ValHistory):
            `ValHistory` instance containing the validation loss curve.
            If not provided, no validation loss curve is plotted.
        agg_label (optional, str):
            Label indicating the aggregation applied to the losses.
            If provided, the labels for the loss curves will appear as
            `Train Loss (agg_label)` and `Val Loss (agg_label)`.
            If `None`, labels are displayed without the `agg_label` prefix.
            Default is `Avg` for average.
        figsize (optional, Tuple[float, float]):
            Figure size in the form of a tuple (width, height).
            If not provided, defaults to `6 * num_hist, 4`,
            where `num_hist` is the number of provided history objects.
    Returns:
        Figure:
            Matplotlib figure displaying the loss curves for training and/or validation.
    '''
    # Setup curve information
    agg_label = '' if agg_label is None else f' ({agg_label})'
    histories, curve_labels, clrs = [], [], []
    if train_history is not None:
        histories.append(train_history)
        curve_labels.append(f'Train Loss{agg_label}')
        clrs.append('tab:blue')
    if val_history is not None:
        histories.append(val_history)
        curve_labels.append(f'Val Loss{agg_label}')
        clrs.append('tab:red')
        
    num_hist = len(histories)
    if num_hist == 0:
        raise ValueError('At least one of train_history and val_history must be provided.')
        
    # Setup figure
    figsize = (6 * num_hist, 4) if figsize is None else figsize
    fig, axes = plt.subplots(ncols = num_hist, nrows = 1, figsize = figsize)
    flat_axes = axes.flatten() if num_hist > 1 else [axes]

    # Plotting
    for i, (ax, history, label, clr) in enumerate(zip(flat_axes, histories, curve_labels, clrs)):
        epochs = history.loss.epochs
        loss = history.loss.values
        ax.plot(epochs, loss, label = label, color = clr)
        
        if i > 0:
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position('right')
            y_rot, y_pad = 270, 20
        else:
            y_rot, y_pad = 90, 0
            
        ax.set_ylabel('Loss', rotation = y_rot, labelpad = y_pad)
        ax.set_xlabel('Epoch')
        ax.legend()
        
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_summary_metrics(
    val_history: ValHistory,
    metric_specs: Dict[str, MetricSpecLike],
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    '''
    Plots summarized versions of validation metric curves.
    Each validation metric is displayed on a separate subplot/panel.

    Note: Each metric specification in `metric_specs`,
          extracts a series of metric values from `val_history.metrics.values`.
          Each value corresponds to a single epoch and is optionally subsetted and/or aggregated.
          The resulting series/curve is expected to only contain scalar values.
    
    Args:
        val_history (ValHistory): 
            `ValHistory` instance containing the validation metrics series to plot.
            These should be contained in the attribute `val_history.metrics.values`.
        metric_specs (Dict[str, MetricSpecLike]):
            Dictionary mapping label names to metric specifications (`MetricSpecLike`)
            indicating what metrics curves to produce and plot.
            Supported types:
                - `MetricSpec` instance.
                - `MetricSpecDict` dictionary.
        nrows (optional, int): 
            Nunber of rows in the figure.
            If not provided, it is set to `math.ceil(num_specs / ncols)`.
            This applies even when `ncols` is not provided, since `ncols` defaults to `math.ceil(math.sqrt(num_specs))`.
            Here, `num_specs = len(metric_spects)`.
        ncols (optional, int): 
            Number of columns in the figure.
            If not provided, but `nrows` is provided, it is set to `math.ceil(num_specs / nrows)`.
            If not provided and `nrows` is not provided, it is set to `math.ceil(math.sqrt(num_specs))`.
            Here, `num_specs=len(metric_specs)`.
        figsize (optional, Tuple[float, float]):
            Figure size in the form of a tuple (width, height).
            If not provided, defaults to `5 * ncols, 5 * nrows`.
    Returns:
        Figure:
            Matplotlib figure displaying the validation metric curves produced from `val_history`.
    '''
    # Setup figure and colors
    num_specs = len(metric_specs)
    fig, axes = make_grid(num_specs, nrows, ncols, figsize)
    flat_axes = axes.flatten() if num_specs > 1 else [axes]
    clrs = sns.color_palette(palette = 'hls', n_colors = num_specs)

    # Get metric history values and epochs
    all_metric_series = val_history.metrics.values
    epochs = val_history.metrics.epochs

    # Plotting
    for ax, (label, spec), clr in zip(flat_axes, metric_specs.items(), clrs):
        spec = format_metric_spec(spec)
        metric_series = select_and_agg_scalar_metric(
            metric_data = all_metric_series,
            key_path = spec.key_path,
            class_idxs = spec.class_idxs,
            agg = spec.agg
        )

        ax.plot(epochs, metric_series, c = clr, label = label)
    
        ax.set_ylabel('Metric')
        ax.set_xlabel('Epoch')
        ax.legend()
    
    for ax in flat_axes[num_specs:]:
        ax.axis('off')

    fig.suptitle('Validation Summary Metrics', fontsize = 40, y = 0.99)
    fig.tight_layout(w_pad = 2, h_pad = 1.5)
    return fig


def plot_class_metrics(
    val_history: ValHistory,
    key_paths: List[str],
    class_names: List[str],
    metric_labels: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title_fontsize: PythonNum = 25
) -> Figure:
    '''
    Plots class-wise validation metric curves.
    Each validation metric is displayed on a separate subplot/panel 
    and each class recieves their own curve within that panel.

    Note: Each key path from `key_paths` is used to extract
          a series of class-wise metric tensors from `val_history.metrics.values`.
          Each tensor corresponds to a single epoch and is expected 
          to have a length equal to the number of classes (i.e. `len(class_names)`).
    
    Args:
        val_history (ValHistory):
            `ValHistory` instance containing the class-wise validation metrics series to plot.
            These should be contained in the attribute `val_history.metrics.values`.
        key_paths (List[str]):
            List of dot-separated key paths within `val_history.metrics.values`.
        class_names (List[str]): 
            List of class names. 
        metric_labels (optional, List[str]):
            Labels for the title of each class-wise metric subplot/panel.
            If provided, must be the same length as `key_paths`.
            If not provided, the key paths in `key_paths` are used as titles.
        nrows (optional, int): 
            Nunber of rows in the figure.
            If not provided, it is set to `math.ceil(num_paths / ncols)`.
            This applies even when `ncols` is not provided, since `ncols` defaults to `math.ceil(math.sqrt(num_paths))`.
            Here, `num_paths = len(key_paths)`.
        ncols (optional, int): 
            Number of columns in the figure.
            If not provided, but `nrows` is provided, it is set to `math.ceil(num_paths / nrows)`.
            If not provided and `nrows` is not provided, it is set to `math.ceil(math.sqrt(num_paths))`.
            Here, `num_paths=len(metric_paths)`.
        figsize (optional, Tuple[float, float]):
            Figure size in the form of a tuple (width, height).
            If not provided, defaults to `5 * ncols, 5 * nrows`.
        title_fontsize (PythonNum): 
            Fontsize for the titles of each subplot/panel. Default is `25`.

    Returns:
        Figure:
            Matplotlib figure displaying the class-wise evaluation metric curves extracted from `val_history`.
    '''
    num_paths = len(key_paths)
    num_classes = len(class_names)

    if metric_labels is None:
        metric_labels = key_paths
    elif len(metric_labels) != num_paths:
        raise ValueError('If metric_labels is provided, it must have the same length as key_paths.')
    
    # Setup figure and colors
    fig, axes = make_grid(num_paths, nrows, ncols, figsize)
    flat_axes = axes.flatten() if num_paths > 1 else [axes]
    class_clrs = sns.color_palette(palette = 'hls', n_colors = num_classes)

    # Get metric history values and epochs
    history_values = val_history.metrics.values
    epochs = val_history.metrics.epochs

    # Plotting
    for ax, key_path, metric_label in zip(flat_axes, key_paths, metric_labels):
        metric_values = nested_extract(history_values, key_path)
        first_val = metric_values[0]
        if isinstance(first_val, torch.Tensor):
            metric_values = torch.stack(metric_values)
            
            k = metric_values.shape[-1]
            ndim = metric_values.ndim
            if k != num_classes:
                raise ValueError(
                    f"Metric at key_path '{key_path}' has {k} classes, "
                    f'but expected {num_classes}.'
                )   
            if ndim != 2:
                raise ValueError(
                    f"Metric at key_path '{key_path}' should be 1D tensors, "
                    f'but got {ndim - 1}D'
                )
        else:
            raise TypeError(
                f"key_path '{key_path}' did not produce a list of tensors. "
                f'Got: {type(first_val)}'
            )
            
        # Plotting per-class metric curves
        for i, (class_name, clr) in enumerate(zip(class_names, class_clrs)):
            class_values = metric_values[:, i].tolist()
            ax.plot(epochs, class_values, c = clr, label = class_name)
        
        ax.set_title(metric_label, fontsize = title_fontsize)
        ax.set_ylabel('Metric')
        ax.set_xlabel('Epoch')
        ax.legend()

    for ax in flat_axes[num_paths:]:
        ax.axis('off')

    fig.suptitle('Validation Class Metrics', fontsize = 40, y = 0.99)
    fig.tight_layout(w_pad = 2, h_pad = 1.5)
    return fig