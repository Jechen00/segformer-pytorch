#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn

from typing import Literal


#####################################
# Loss Classes
#####################################
class FocalLoss(nn.Module):
    '''
    Implements focal loss for binary classification, 
    as described in the paper: https://arxiv.org/pdf/1708.02002
    
    Assuming p = probability of positive target P(y = 1):
        - For positive targets (y = 1), the loss term is `-alpha * (1 - p)^gamma * log(p)`.
        - For negative targets (y = 0), the loss term is `-alpha * p^gamma * log(1 - p)`.

    Note: This implementation relies on `torch.nn.CrossEntropyLoss`, 
    so the expected inputs are model logits (not predictions) and target labels.

    Args:
        alpha (float): Alpha parameter of the focal loss.
                       This determines how much the loss will weigh positive samples (y = 1) 
                       compared to negative  (y = 0).
                       Specifically, loss terms for positive samples are weighed by `alpha`, 
                       while loss terms for negative samples are weighted by `1 - alpha`.
                       If `alpha > 0.5`, positive samples contribute more to the loss.
                       If `alpha < 0.5`, negative samples contribute more to the loss.
                       Default is `0.5`, which weights positive and negative samples equally.
        gamma (float): Gamma parameter of the focal loss.
                       This determines how much the loss downweights samples
                       that are predicted correctly with high probability by the model.
                       Increasing `gamma` more strongly reduces the loss contribution of 
                       these correctly-predicted samples.
                       If `gamma = 0`, this will default back to the behavior of 
                       a standard binary cross entropy loss. 
                       Default is `2`, following the original paper.
        ignore_index (int): Index/label of samples to ignore during loss computation.
                            Default is `-100`, following PyTorch conventions.
        reduction (Literal['mean', 'sum', 'none']): The reduction to apply to the loss.
            - 'mean': Returns the mean loss across all valid (non-ignored) samples.
            - 'sum': Returns the summed loss across all valid (non-ignored) samples.
            - 'none': Returns the loss with no reduction.
    '''
    def __init__(
        self, 
        alpha: float = 0.5, 
        gamma: float = 2.0, 
        ignore_index: int = -100, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        self.ce = nn.CrossEntropyLoss(
            ignore_index = ignore_index, 
            reduction = 'none'
        )
        
    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        '''
        Computes the focal loss between predictions and targets.
        For more information on input shapes, see:
            https://docs.pytorch.org/docs/2.12/generated/torch.nn.CrossEntropyLoss.html

        Args:
            logits (torch.Tensor): Model logits.
                                   These will be converted to predictions using softmax.
                                   Supported shapes:
                                        - 1D tensor: (num_classes,)
                                        - 2D tensor: (batch_size, num_classes)
                                        - nD tensor: (batch_size, num_classes, d_1, ..., d_k),
                                                     where k = n-2
            targs (torch.Tensor): Target labels corresponding to `logits`.
                                  The shape must match `logits`, excluding the `num_classes` dimension.
                                  Supported shapes:
                                        - scalar tensor
                                        - 1D tensor: (batch_size,)
                                        - (n-1)D tensor:  (batch_size, d_1, ..., d_k)
        Returns:
            torch.Tensor: The computed focal loss.
                          If `self.reduction='none'`, this will be the same shape as `targs`.
                          Otherwise, this will be a scalar.
        '''
        ce_loss = self.ce(logits, targs)
        p_t = torch.exp(-ce_loss)

        focal_loss = (1 - p_t)**self.gamma * ce_loss

        valid_mask = (targs != self.ignore_index)
        valid_targs = targs[valid_mask]

        alpha_t = torch.where(valid_targs == 0, 1 - self.alpha, self.alpha)
        focal_loss[valid_mask] *= alpha_t # Multiple valid elements by alpha_t

        # Aggregate loss
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'sum':
            return focal_loss.sum()
        elif self.reduction == 'mean':
            return focal_loss.sum() / valid_mask.sum()