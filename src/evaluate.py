#####################################
# Imports & Dependencies
#####################################
import torch

from typing import Literal

#####################################
# Functions
#####################################
def calc_accuracy(
        logits: torch.Tensor, 
        targs: torch.Tensor, 
        reduction: Literal['sum', 'mean'] = 'mean'
    ) -> float:
    '''
    Computes accuracy from model logits and target class indices.
    Logits are processed into predicted class indices using argmax on the class dimension.

    This function supports classification and segmentation:
        - Classification: accuracy is computed over batch samples.
        - Segmentation: accuracy is computed over all pixels across the batch.

    Args:
        logits (torch.Tensor): Logit tensor produced by a classification or segmentation model.
                               Shape depends on the task:
                                    - Classification: (batch_size, num_classes)
                                    - Segmentation: (batch_size, num_classes, height, width)
        targs (torch.Tensor): Target tensor containing ground truth class indices.
                              Shape depends on the task:
                                    - Classification: (batch_size,)
                                    - Segmentation: (batch_size, height, width)
        reduction ('sum' or 'mean'): The reduction to apply to the output.
                                        - 'sum': Returns total number of correct predictions.
                                        - 'mean': Returns accuracy, normalized by total number of elements.
                                     Default is 'mean'.
                                     
    Returns:
        float: Accuracy between logits.argmax(1) and targs.
    '''
    # Classification preds shape: (batch_size,)
    # Segmentation preds shape: (batch_size, height, width)
    preds = logits.argmax(dim = 1)
    correct = (preds == targs).sum().item()

    if reduction == 'mean':
        return correct / preds.numel()
    else:
        return correct