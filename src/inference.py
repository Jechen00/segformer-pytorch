#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torchvision.transforms import v2

from PIL import Image
from typing import Union, List, Optional

from src.data_setup.types import SampleDict, Sample, MultiSamples
from src.tensor_shapes import (
    ensure_batched, _validate_same_shape, _validate_ndim
)
from src.utils import transpose_list_dict


#####################################
# Inference Functions
#####################################
def preprocess_and_predict(
    model: nn.Module,
    samps: Union[Sample, MultiSamples], 
    transforms: Optional[v2.Compose] = None,
    memory_format: Optional[torch.memory_format] = None
) -> torch.Tensor:
    '''
    Preprocesses samples and computes predictions on their images.
    Assumes model outputs logits with class dimension at `dim = 1`.

    This is done by calling:
        1. preprocess_imgs
        2. predict
    '''
    imgs = preprocess_imgs(samps, transforms)
    preds = predict(model, imgs, memory_format)
    return preds


def predict(
    model: nn.Module, 
    imgs: torch.Tensor, 
    memory_format: Optional[torch.memory_format] = None
) -> torch.Tensor:
    '''
    Computes predictions for a image classification/segmentation model.
    Assumes model outputs logits with class dimension at `dim = 1`.

    Args:
        model (nn.Module): 
            The image classification/segmentation model,
        imgs (torch.Tensor): 
            The images to predict on.
            This should be a tensor of shape `(batch_size, channels, height, width)`.
        memory_format (optional, torch.memory_format): 
            The memory format to convert `imgs` to before predicting.
            This should ideally be the same memory format as `model`, but it is not required.
            If not provided, no memory format conversion is applied.
    Returns:
        torch.Tensor: 
            Model predictions for `imgs`, with shape depending on the type of model.
            Generally:
                - Image Classification: `(batch_size,)`
                - Segmentation: `(batch_size, height, width)`

    '''
    was_training = model.training

    # Send imgs to device of model
    device = next(model.parameters()).device
    imgs = imgs.to(device)

    # Send imgs to memory_format if provided
    if memory_format is not None:
        imgs = imgs.to(memory_format = memory_format)

    # Get predictions
    model.eval()
    with torch.inference_mode():
        logits = model(imgs)
        preds = logits.argmax(dim = 1).cpu()
        
    if was_training:
        model.train()

    return preds


def preprocess_imgs(
    samps: Union[Sample, MultiSamples], 
    transforms: Optional[v2.Compose] = None
) -> torch.Tensor:
    '''
    Preprocesses the image(s) in a sample or multiple samples, preparing them for input into `predict`.

    This involves:
        1. Formatting input into a sample dictionary or list of sample dictionaries
        2. Optionally applying transforms
        3. Extracting all images
        5. Stacking images into a batched tensor

    Note: After optional image transforms, the images must be tensors of the same shape.
          If the images are contained in a list, each processed image must be a 3D tensor.
          Otherwise, the processed image must be a 3D or 4D tensor.
          If image transforms were not provided, make sure that the input samples 
          already contain image tensors with the expected shape and number of dimensions.

    Args:
        samps (Union[Sample, MultiSamples]): 
            Sample or multiple samples containing image information.
            Supports:
                - A single image. This is a PIL image or 3D tensor of shape `(channels, height, width)`
                - A single-sample dictionary, where the 'image' key contains a single image
                - A list of single images
                - A multi-sample dictionary, where the 'image' key contains a list of images
                - A list of single-sample dictionaries
                - A batched 4D tensor of shape `(batch_size, channels, height, width)`

        transforms (optional, v2.Compose): 
            Transforms applied to `samps` before extracting images.
            These transforms should be compatible with a sample dictionary,
            where images are stored under the `image` key.

    Returns:
        torch.Tensor: 
            A batched tensor of shape `(batch_size, channels, height, width)`.
            If the input is a single sample, `batch_size` is 1.
    '''
    samps = _format_samps(samps)
    is_list = isinstance(samps, list)

    # Apply image transforms if provided
    if transforms is not None:
        if is_list:
            samps = [transforms(s) for s in samps]
        else:
            samps = transforms(samps)

    # Extract images from samps and batch
    # samps is a list of sample dictionaries
    if is_list:
        imgs = [s['image'] for s in samps]
        _validate_same_shape(imgs, context_name = 'processed images')
        _validate_ndim(imgs[0], ndim = 3, context_name = 'processed images')
        return torch.stack(imgs)
    
    # samps is a sample dictionary
    imgs = samps['image']
    if not isinstance(imgs, torch.Tensor):
        raise ValueError(
            'Expected processed images to be a 3D or 4D tensor. '
            f'Got: {type(imgs)}'
        )
    
    return ensure_batched(imgs, unbatched_ndim = 3, context_name = 'processed images')


#####################################
# Formatting Functions
#####################################
def _format_samps(
    samps: Union[Sample, MultiSamples]
) -> Union[SampleDict, List[SampleDict]]:
    '''
    Formats image samples for processing in `preprocess_imgs`.
    This converts all samples into either a sample dictionary or a list of sample dictionaries.

    Note: The value contained in the `image` key of each sample dictionary is preserved. 
    This means PIL images remain PIL images, 
    and tensors keep their original shape (including any batch dimension).

    Args:
        samps (Union[Sample, MultiSamples]): 
            Sample or multiple samples containing image information.
            Supports:
                - A single image. This is a PIL image or 3D tensor of shape `(channels, height, width)`
                - A single-sample dictionary, where the 'image' key contains a single image
                - A list of single images
                - A multi-sample dictionary, where the 'image' key contains a list of images
                - A list of single-sample dictionaries
                - A batched 4D tensor of shape `(batch_size, channels, height, width)`

    Returns:
        Union[SampleDict, List[SampleDict]]: 
            Formatted samples.
            If `samps` is a sample dictionary, it is returned unchanged.
            If `samps` is a PIL image or tensor, it is wrapped in a sample dictionary.
            Otherwise, `samps` is converted into a list of sample dictionaries.
    '''
    # Dictionary input
    if isinstance(samps, dict):
        img = samps['image']
        if isinstance(img, list):
            return transpose_list_dict(samps, mode = 'to_rows') # List of sample dictionaries
        
        elif not isinstance(img, (Image.Image, torch.Tensor)):
            raise TypeError(
                "Expected the 'image' key of dictionary inputs to contain "
                'either a PIL image, a tensor, or a list of PIL images/tensors.'
                f'Got: {type(samps)}'
            )
        
        return samps # Sample dictionary

    # List input
    elif isinstance(samps, list):
        # Format each element into a sample dictionary
        formatted_list = []
        for samp in samps:
            if isinstance(samp, (Image.Image, torch.Tensor)):
                formatted_list.append({'image': samp})
            elif isinstance(samp, dict):
                formatted_list.append(samp)
            else:
                raise TypeError(
                    'Expected list inputs to only contain '
                    'PIL images, tensors, or dictionaries'
                    f'Got: {type(samp)}'
                )
            
        return formatted_list # List of sample dictionaries

    # Image or tensor input
    elif isinstance(samps, (Image.Image, torch.Tensor)):
        return {'image': samps} # Sample dictionary

    else:
        raise TypeError(
            'Expected samps to be a PIL image, tensor, list, or dictionary. '
            f'Got: {type(samps)}'
        )