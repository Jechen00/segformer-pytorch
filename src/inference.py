#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torchvision.transforms import v2

from PIL import Image
from typing import Union, Optional

from src.ml_types import Sample, BatchedSamples
from src.utils import transpose_list_dict, extract_imgs, check_tensor_shapes


#####################################
# Functions
#####################################
def preprocess_and_predict(
    model: nn.Module,
    samps: Union[Sample, BatchedSamples], 
    img_transforms: Optional[v2.Compose] = None
) -> torch.Tensor:
    '''
    Preprocesses samples and computes predictions for them.
    Assumes model outputs logits with class dimension at `dim = 1`.

    This is done by calling:
        1. preprocess_imgs
        2. predict
    '''
    imgs = preprocess_imgs(samps, img_transforms)
    preds = predict(model, imgs)
    return preds


def predict(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    '''
    Computes predictions for a image classification/segmentation model.
    Assumes model outputs logits with class dimension at `dim = 1`.

    Args:
        model (nn.Module): The image classification/segmentation model,
        imgs (torch.Tensor): The images to predict on.
                             This should be a tensor of shape `(batch_size, channels, height, width)`.
    Returns:
        torch.Tensor: Model predictions for `imgs`, with shape depending on the type of model.
                      Generally:
                        - Image Classification: `(batch_size,)`
                        - Segmentation: `(batch_size, height, width)`

    '''
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    with torch.inference_mode():
        logits = model(imgs.to(device))
        preds = logits.argmax(dim = 1).cpu()
        
    if was_training:
        model.train()

    return preds


def preprocess_imgs(
    samps: Union[Sample, BatchedSamples], 
    img_transforms: Optional[v2.Compose] = None
) -> torch.Tensor:
    '''
    Preprocesses the image(s) in a sample or batch of samples, preparing them for input into `predict`.

    This involves:
        1. Normalizing input format
        2. Optionalluy applying image transformations
        3. Extracting all images
        4. Checking that all images are tensors of the same shape
        5. Stacking images into a batched tensor

    Args:
        samps (Union[Sample, BatchedSamples]): Sample or batch of samples containing image information.
            Supports:
                - A single image (PIL image or tensor)
                - A single-sample dictionary, where the 'image' key contains a single image
                - A list of images (PIL image or tensor)
                - A batched-sample dictionary, where the 'image' key contains a list of images
                - A collated tensor, e.g. of shape (batch_size, channels, height, width)

        img_transforms (optional, v2.Compose): Image transfromations applied to `samps` before extracting images.

    Returns:
        torch.Tensor: A batched tensor of shape `(batch_size, channels, height, width)`.
                      If the input is a single sample, `batch_size` is 1.
    '''
    # Check input datatype and normalize
    if isinstance(samps, dict):
        # Transpose to a list of dictionaries
        # This is in case transforms don't support a dictionary of lists
        if isinstance(samps['image'], list):
            samps = transpose_list_dict(samps, mode = 'to_rows')

    elif isinstance(samps, list):
        if len(samps) == 0:
            raise ValueError('List inputs must be non-empty.')

    elif not isinstance(samps, (Image.Image, torch.Tensor)):
        raise TypeError(
            f'Expected samps to be a PIL image, tensor, list, or dictionary. Got: {type(samps)}'
        )

    # Apply image transforms if provided
    if img_transforms is not None:
        if isinstance(samps, list):
            samps = [img_transforms(s) for s in samps]
        else:
            samps = img_transforms(samps)

    # Extract images from samps
    imgs = extract_imgs(samps)

    # Check images are tensors and shapes are all the same
    imgs_list = imgs if isinstance(imgs, list) else [imgs]
    check_tensor_shapes(imgs_list)

    # Ensure imgs has a batch dimension
        # Triggers when imgs is a list of samples or samps was a single-sample input
    if (imgs_list[0].ndim == 3):
        imgs = torch.stack(imgs_list)
        
    return imgs