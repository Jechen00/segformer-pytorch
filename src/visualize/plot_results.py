#####################################
# Imports & Dependencies
#####################################
import torch

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, Tuple, List

from src.logging.history import TrainHistory, ValHistory
from src.visualize.layout import make_grid
from src.utils import nested_extract, apply_agg
from src.ml_types import MetricLogFields


#####################################
# Functions
#####################################
def plot_loss(
    train_history: Optional[TrainHistory],
    val_history: Optional[ValHistory],
    agg_label: Optional[str] = 'Avg',
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
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
    fields: MetricLogFields,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    # Setup figure and colors
    num_fields = len(fields)
    fig, axes = make_grid(num_fields, nrows, ncols, figsize)
    flat_axes = axes.flatten() if num_fields > 1 else [axes]
    clrs = sns.color_palette(palette = 'hls', n_colors = num_fields)

    # Get metric history values and epochs
    history_values = val_history.metrics.values
    epochs = val_history.metrics.epochs

    # Plotting
    for ax, field, clr in zip(flat_axes, fields, clrs):
        if isinstance(field, tuple):
            key_path, agg = field
        else:
            key_path, agg = field, 'mean'

        metric_values = nested_extract(history_values, key_path)

        if isinstance(metric_values[0], torch.Tensor):
            metric_values = [apply_agg(val, agg) for val in metric_values]
            curve_label = f'{key_path} ({agg})'
        else:
            curve_label = key_path

        ax.plot(epochs, metric_values, c = clr, label = curve_label)
    
        ax.set_ylabel('Metric')
        ax.set_xlabel('Epoch')
        ax.legend()
    
    for ax in flat_axes[num_fields:]:
        ax.axis('off')

    fig.suptitle('Validation Summary Metrics', fontsize = 40, y = 0.99)
    fig.tight_layout(w_pad = 2, h_pad = 1.5)
    return fig


def plot_class_metrics(
    val_history: ValHistory,
    key_paths: List[str],
    class_names: List[str],
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None
) -> Figure:
    num_paths = len(key_paths)
    num_classes = len(class_names)

    # Setup figure and colors
    fig, axes = make_grid(num_paths, nrows, ncols, figsize)
    flat_axes = axes.flatten() if num_paths > 1 else [axes]
    class_clrs = sns.color_palette(palette = 'hls', n_colors = num_classes)

    # Get metric history values and epochs
    history_values = val_history.metrics.values
    epochs = val_history.metrics.epochs

    # Plotting
    for ax, key_path in zip(flat_axes, key_paths):
        metric_values = nested_extract(history_values, key_path)
        first_val = metric_values[0]
        if isinstance(first_val, torch.Tensor):
            metric_values = torch.stack(metric_values)
            
            k = metric_values.shape[-1]
            ndim = metric_values.ndim
            if k != num_classes:
                raise ValueError(
                    f'Metric at key_path {key_path} has {k} classes, '
                    f'but expected {num_classes}.'
                )   
            if ndim != 2:
                raise ValueError(
                    f'Metric at key_path {key_path} should be 1D tensors, '
                    f'but got {ndim - 1}D'
                )
        else:
            raise TypeError(
                f'key_path {key_path} did not produce a list of tensors. '
                f'Got: {type(first_val)}'
            )
            
        # Plotting per-class metric curves
        for i, (class_name, clr) in enumerate(zip(class_names, class_clrs)):
            class_values = metric_values[:, i].tolist()
            ax.plot(epochs, class_values, c = clr, label = class_name)
        
        ax.set_title(key_path, fontsize = 25)
        ax.set_ylabel('Metric')
        ax.set_xlabel('Epoch')
        ax.legend()

    for ax in flat_axes[num_paths:]:
        ax.axis('off')

    fig.suptitle('Validation Class Metrics', fontsize = 40, y = 0.99)
    fig.tight_layout(w_pad = 2, h_pad = 1.5)
    return fig