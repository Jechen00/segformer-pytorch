#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torchvision.transforms import v2

from PIL import Image
from matplotlib.figure import Figure
from typing import List, Optional, Union, Tuple

from src.inference import preprocess_and_predict
from src.visualize.layout import make_grid
from src.ml_types import ImageInput, ImageLabel


#####################################
# Image Classification Functions
#####################################
def plot_cls_preds(
    model: nn.Module,
    class_names: List[str],
    imgs: Union[ImageInput, List[ImageInput]],
    labels: Union[ImageLabel, List[ImageLabel]] = None,
    img_transforms: Optional[v2.Compose] = None,
    memory_format: Optional[torch.memory_format] = None,
    nrows: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title_fontsize: float = 18,
) -> Figure:
    '''
    Note: `memory_format` will only convert `imgs`.
          Ideally, `model` and `imgs` should be on the same memory format.
    '''
    # Normalize single-sample images
    if ((isinstance(imgs, torch.Tensor) and imgs.ndim == 3)
        or isinstance(imgs, Image.Image)):
        imgs = [imgs]

    num_imgs = len(imgs)

    # Normalize targets to lists of integers
    if labels is None:
        labels = [None] * len(imgs)
    else:
        labels = make_labels_list(labels)

    # Setup figure
    fig, axes = make_grid(num_imgs, nrows, ncols, figsize)
    flat_axes = axes.flatten() if num_imgs > 1 else [axes]

    # Predictions
    preds = preprocess_and_predict(model, imgs, img_transforms, memory_format)

    # Plotting
    for ax, img, label, pred in zip(flat_axes, imgs, labels, preds):
        # Plot image
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
        ax.imshow(img)

        # Make title
        title = f'Pred: {class_names[pred]}'
        clr = 'k'
        if label is not None:
            title += f'\nTarg: {class_names[label]}'
            clr = 'g' if (pred == label) else 'r'

        ax.set_title(title, color = clr, fontsize = title_fontsize)
        ax.axis('off')

    for ax in flat_axes[num_imgs:]:
        ax.axis('off')

    fig.tight_layout(h_pad = 1.5)
    return fig


def make_labels_list(labels: Union[ImageLabel, List[ImageLabel]]) -> List[int]:
    # Normalize int and tensor inputs
    if isinstance(labels, int):
        return [labels]

    elif isinstance(labels, torch.Tensor):
        if labels.ndim > 1:
            raise ValueError('If labels is a tensor, it must have ndim <= 1.')
        return labels.int().flatten().tolist()

    elif not isinstance(labels, list):
        raise TypeError(
            'If provided, labels must be a tensor, integer, or list of integers.'
            f'Got: {type(labels)}'
        )
        
    # Normalize list inputs
    norm_labels = []
    for label in labels:
        if isinstance(label, int):
            norm_labels.append(label)
        
        elif isinstance(label, torch.Tensor):
            if label.numel() != 1:
                raise ValueError(
                    'If labels is a list, all tensor elements must be scalars. '
                    f'Got: label.numel() = {label.numel()}'
                )
            norm_labels.append(label.int().item())
            
        else:
            raise TypeError(
                'If labels is a list, all elements must be a tensor or integer. '
                f'Got: {type(label)}'
            )
    return norm_labels


#####################################
# Segmentation Functions
#####################################
def plot_seg_preds_collage():
    return

def plot_seg_preds_overlap():
    return
