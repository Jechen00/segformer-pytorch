#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
import torch.nn.functional as F

from typing import Literal, Optional, Dict


#####################################
# Individual Loss Classes
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
                       This determines how much the loss will weigh positive elements (`y = 1`) 
                       compared to negative  (`y = 0`).
                       Specifically, loss terms for positive elements are weighed by `alpha`, 
                       while loss terms for negative elements are weighted by `1 - alpha`.
                       If `alpha > 0.5`, positive elements contribute more to the loss.
                       If `alpha < 0.5`, negative elements contribute more to the loss.
                       Default is `0.5`, which weights positive and negative elements equally.
        gamma (float): Gamma parameter of the focal loss.
                       This determines how much the loss downweights elements
                       that are predicted correctly with high probability by the model.
                       Increasing `gamma` more strongly reduces the loss contribution of 
                       these correctly-predicted elements.
                       If `gamma = 0`, this will default back to the behavior of 
                       a standard binary cross entropy loss. 
                       Default is `2`, following the original paper.
        ignore_idx (int): Target index/label for elements to ignore during loss computation.
                          Default is `-100`, following PyTorch conventions.
        reduction (Literal['mean', 'sum', 'none']): The reduction to apply to the loss.
            - 'mean': Computes the mean loss across all valid (non-ignored) elements.
            - 'sum': Computes the summed loss across all valid (non-ignored) elements.
            - 'none': Computes the loss without reducing across elements.
            Default is 'mean'.
    '''
    def __init__(
        self, 
        alpha: float = 0.5, 
        gamma: float = 2.0, 
        ignore_idx: int = -100, 
        reduction: Literal['mean', 'sum', 'none'] = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_idx = ignore_idx
        self.reduction = reduction
        
        self.ce = nn.CrossEntropyLoss(
            ignore_index = ignore_idx, 
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
                                        - 1D tensor: `(num_classes,)`
                                        - 2D tensor: `(batch_size, num_classes)`
                                        - nD tensor: `(batch_size, num_classes, d_1, ..., d_k)`,
                                                     where `k = n-2`
            targs (torch.Tensor): Target tensor with the same batch and spatial dimensions as `logits`.
                                  Supported shapes:
                                        - scalar tensor
                                        - 1D tensor: `(batch_size,)`
                                        - (n-1)D tensor: `(batch_size, d_1, ..., d_k)`
        Returns:
            torch.Tensor: 
              The computed focal loss.
              The shape depends on the reduction.
                  - If `reduction='mean'`: Returns a scalar tensor.
                  - If `reduction='sum'`: Returns a scalar tensor.
                  - If `reduction='none'`: Returns a tensor with the same shape as `targs`.
        '''
        ce_loss = self.ce(logits, targs)
        p_t = torch.exp(-ce_loss)

        focal_loss = (1 - p_t)**self.gamma * ce_loss

        valid_mask = (targs != self.ignore_idx)
        valid_targs = targs[valid_mask]

        alpha_t = torch.where(valid_targs == 0, 1 - self.alpha, self.alpha)
        focal_loss[valid_mask] *= alpha_t # Multiply valid elements by alpha_t

        # Aggregate loss
        if self.reduction == 'none':
            return focal_loss # Scalar
        elif self.reduction == 'sum':
            return focal_loss.sum() # Scalar
        elif self.reduction == 'mean':
            return focal_loss.sum() / valid_mask.sum() # Shape same as targs 


class DiceLoss(nn.Module):
    '''
    Implements multi-class Dice loss for semantic segmentation.
    
    Note: the standard binary Dice loss can be obtained by setting `num_classes=2` 
    and `exclude_bg_idx` to the background index.
    
    Note: This only supports batched tensors as input.
          Moreover, it requires raw model logits and target class indices (not one-hot).
          Target class indices must be in the range `[0, num_classes - 1]`,
          with the only exception being the `ignore_idx` (if specified).
    
    Args:
        num_classes (int): Number of classes.
        ignore_idx (int): Target index/label for elements to ignore during loss computation.
                          Default is `-100`, following PyTorch conventions.
        exclude_bg_idx (optional, int): Index of the background class.
    `       If provided, a Dice loss is not computed for the background class.
            When `reduction` is 'mean' or 'sum', this excludes the background class from the aggregation.
            When`reduction` is `none`, this replaces the background class with `torch.nan` for all batch samples.
            Default is `None`, meaning that the background class is included.`
        reduction (Literal['mean', 'sum', 'none']): The reduction to apply to the loss.
            - 'mean': Computes the mean loss across all included classes 
                     and valid batch samples (samples with at least one non-ignored element).
            - 'sum': Computes the summed loss across all included classes 
                     and valid batch samples (samples with at least one non-ignored element).
            - 'none': Computes the loss without any reducing across classes and batch samples.
            Default is `mean`.
        eps (float): A small constant used to prevent numerical errors (e.g. divide by zero).
                     Default is `1e-6`.
    '''
    def __init__(
        self, 
        num_classes: int, 
        ignore_idx: int = -100,
        exclude_bg_idx: Optional[int] = None,
        reduction: Literal['mean', 'sum', 'none'] = 'mean', 
        eps: float = 1e-6
    ):
        if exclude_bg_idx is not None:
            if not (0 <= exclude_bg_idx < num_classes):
                raise ValueError(
                    'If exclude_bg_idx is provided, it must be an integer in the range [0, num_classes - 1].'
                )
                
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        
        super().__init__()
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
        self.exclude_bg_idx = exclude_bg_idx
        self.reduction = reduction
        self.eps = eps
        
    def forward(self, logits: torch.Tensor, targs: torch.Tensor) -> torch.Tensor:
        '''
        Computes multi-class Dice loss between predictions and targets.
        
        Args:
            logits (torch.Tensor): Model logits of shape `(batch_size, num_classes, d_1, ..., d_k)`.
            targs (torch.Tensor): Target tensor with the same batch and spatial dimensions as `logits`.
                                  This must contain integer class indices rather than one-hot labels.
                                  Shape must be `(batch_size, d_1, ..., d_k)`.
                                  
        Returns:
            torch.Tensor: The computed multi-class Dice loss. The shape depends on the reduction.
                  - If `reduction='mean'`: Returns a scalar tensor.
                  - If `reduction='sum'`: Returns a scalar tensor.
                  - If `reduction='none'`: Returns a tensor of shape `(batch_size, num_classes)`.
        '''
        targs = targs.flatten(1).clone() # Shape: (batch_size, num_pixels)
        preds = F.softmax(logits, dim = 1).flatten(2) # Shape: (batch_size, num_classes, num_pixels)
        
        valid_mask = (targs != self.ignore_idx) # Mask for valid pixels
        num_valid_samps = (valid_mask.sum(-1) != 0).sum() # Number of samples with at least one valid pixel

        # Temporarily sets ignore indices to 0 to allow one-hot
        targs[~valid_mask] = 0

        # Shape: (batch_size, num_classes, num_pixels)
        targs = F.one_hot(targs, num_classes = self.num_classes).transpose(2, 1)

        # Remove background index from calculations if needed
        if self.exclude_bg_idx is not None:
            fg_class_idxs = torch.arange(self.num_classes) != self.exclude_bg_idx
            targs = targs[:, fg_class_idxs]
            preds = preds[:, fg_class_idxs]

        # Set ignore pixels to zero everywhere
            # This ensures no contribution to Dice coefficient
        valid_mask = valid_mask.unsqueeze(1) # Shape: (batch_size, 1, num_pixels)
        preds = preds * valid_mask
        targs = targs * valid_mask

        # Compute Dice loss
        dice_num = 2 * (targs * preds).sum(dim = -1) + self.eps # Shape: (batch_size, num_valid_classes)
        dice_denom = (targs + preds).sum(dim = -1) + self.eps # Shape: (batch_size, num_valid_classes)

        dice_loss = 1 - (dice_num / dice_denom) # Shape: (batch_size, num_valid_classes)

        if self.reduction == 'mean':
            return dice_loss.mean(-1).sum() / (num_valid_samps + self.eps) # Scalar
        elif self.reduction == 'sum':
            return dice_loss.sum() # Scalar
        elif self.reduction == 'none':
            if self.exclude_bg_idx is not None:
                # Insert background class loss as NaN values
                loss = torch.full(
                    (dice_loss.shape[0], self.num_classes),
                    fill_value = torch.nan,
                    dtype = dice_loss.dtype,
                    device = dice_loss.device
                )

                loss[:, fg_class_idxs] = dice_loss
                return loss # (batch_size, num_classes)
            return dice_loss # (batch_size, num_classes)
        

#####################################
# Combined Loss Classes
#####################################
class CEDiceLoss(nn.Module):
    def __init__(
        self, 
        num_classes: int, 
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0,
        ignore_idx: int = -100,
        exclude_bg_idx: Optional[int] = None,
        eps: float = 1e-6
    ):
        '''
        Implements CE-Dice loss for semantic segmentation.
        This is a weighted sum of the mean CE and mean Dice losses:
            `lambda_ce * mean_ce_loss + lambda_dice * mean_dice_loss`

        Note: This only supports batched tensors as input.
              Moreover, it requires raw model logits and target class indices (not one-hot).
              Target class indices must be in the range `[0, num_classes - 1]`,
              with the only exception being the `ignore_idx` (if specified).
        
        Args:
            num_classes (int): Number of classes.
            lambda_dice (float): Weight of the Dice loss. Default is `1.0`.
            lambda_ce (float): Weight of the CE loss. Default is `1.0`.
            ignore_idx (int): Target index/label for elements to ignore during loss computation.
                              Default is `-100`, following PyTorch conventions.
            exclude_bg_idx (optional, int): Index of the background class.
                                            If provided, a Dice loss is not computed for the background class
                                            and the class is not included when computing the mean Dice loss.
                                            This does not affect the CE loss.
                                            Default is `None`, meaning that the background class is included.
            eps (float): A small constant used to prevent numerical errors  in the Dice loss (e.g. divide by zero).
                         Default is `1e-6`.
        '''
        if exclude_bg_idx is not None:
            if not (0 <= exclude_bg_idx < num_classes):
                raise ValueError(
                    'If exclude_bg_idx is provided, it must be an integer in the range [0, num_classes - 1].'
                )
        
        super().__init__()
        self.num_classes = num_classes
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.ignore_idx = ignore_idx
        self.exclude_bg_idx = exclude_bg_idx
        self.eps = eps

        self.ce = nn.CrossEntropyLoss(
            ignore_index = ignore_idx, 
            reduction = 'mean'
        )

        self.dice = DiceLoss(
            num_classes = num_classes,
            ignore_idx = ignore_idx,
            exclude_bg_idx = exclude_bg_idx,
            reduction = 'mean',
            eps = eps
        )

    def forward(
        self, 
        logits: torch.Tensor, 
        targs: torch.Tensor
    ) -> torch.Tensor:
        '''
        Computes the CE-Dice loss between predictions and targets.
        Args:
            logits (torch.Tensor): Model logits of shape `(batch_size, num_classes, d_1, ..., d_k)`.
            targs (torch.Tensor): Target tensor with the same batch and spatial dimensions as `logits`.
                                  This must contain integer class indices rather than one-hot labels.
                                  Shape must be `(batch_size, d_1, ..., d_k)`.
                                  
        Returns:
            torch.Tensor: The CE-Dice loss as a scalar tensor.
        '''
        ce_loss = self.ce(logits, targs)
        dice_loss = self.dice(logits, targs)
        return (self.lambda_ce * ce_loss) + (self.lambda_dice * dice_loss)
    
    def forward_as_components(
        self, 
        logits: torch.Tensor, 
        targs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        '''
        Computes CE-Dice loss and returns a dictionary containing
        each loss component and the total loss.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'ce' (torch.Tensor): The mean CE loss component (unweighted).
                - 'dice' (torch.Tensor): The mean Dice loss component (unweighted).
                - 'total' (torch.Tensor): Weighted sum of mean CE and mean Dice losses.
        '''
        ce_loss = self.ce(logits, targs)
        dice_loss = self.dice(logits, targs)
        tot_loss = (self.lambda_ce * ce_loss) + (self.lambda_dice * dice_loss)

        return {
            'ce': ce_loss,
            'dice': dice_loss,
            'total': tot_loss
        }
    

class FocalDiceLoss(nn.Module):
    def __init__(
        self, 
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
        alpha_focal: float = 0.5, 
        gamma_focal: float = 2.0, 
        ignore_idx: int = -100,
        exclude_bg_idx: Optional[int] = None,
        eps: float = 1e-6
    ):
        '''
        Implements Focal-Dice loss for **binary** semantic segmentation.
        This is a weighted sum of the mean Focal and mean Dice losses:
            `lambda_focal * mean_focal_loss + lambda_dice * mean_dice_loss`

        Note: This only supports batched tensors as input.
              Moreover, it requires raw model logits and target class indices (not one-hot).
              Target class indices must be in the range `[0, num_classes - 1]`,
              with the only exception being the `ignore_idx` (if specified).
        
        Args:
            lambda_dice (float): Weight of the Dice loss. Default is `1.0`.
            lambda_focal (float): Weight of the Focal loss. Default is `1.0`.
            alpha_focal (float): Alpha parameter of the Focal loss. See `FocalLoss` for details.
                                 Default is `0.5`.
            gamma_focal (float): Gamma parameter of the Focal loss. See `FocalLoss` for details.
                                 Default is `2.0`.
            ignore_idx (int): Target index/label for elements to ignore during loss computation.
                              Default is `-100`, following PyTorch conventions.
            exclude_bg_idx (optional, int): Index of the background class.
                                            If provided, a Dice loss is not computed for the background class
                                            and the class is not included when computing the mean Dice loss.
                                            This does not affect the CE loss.
                                            Default is `None`, meaning that the background class is included.
            eps (float): A small constant used to prevent numerical errors  in the Dice loss (e.g. divide by zero).
                         Default is `1e-6`.
        '''
        if exclude_bg_idx is not None:
            if not (0 <= exclude_bg_idx < 2):
                raise ValueError(
                    'If exclude_bg_idx is provided, it must be an integer in the range [0, 1].'
                )
        
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.alpha_focal = alpha_focal
        self.gamma_focal = gamma_focal
        self.ignore_idx = ignore_idx
        self.exclude_bg_idx = exclude_bg_idx
        self.eps = eps

        self.focal = FocalLoss(
            alpha = alpha_focal,
            gamma = gamma_focal,
            ignore_idx = ignore_idx, 
            reduction = 'mean'
        )

        self.dice = DiceLoss(
            num_classes = 2, # This implementation is for binary segmentation
            ignore_idx = ignore_idx,
            exclude_bg_idx = exclude_bg_idx,
            reduction = 'mean',
            eps = eps
        )

    def forward(
        self, 
        logits: torch.Tensor, 
        targs: torch.Tensor
    ) -> torch.Tensor:
        '''
        Computes the Focal-Dice loss between predictions and targets.
        Args:
            logits (torch.Tensor): Model logits of shape `(batch_size, num_classes, d_1, ..., d_k)`.
            targs (torch.Tensor): Target tensor with the same batch and spatial dimensions as `logits`.
                                  This must contain integer class indices rather than one-hot labels.
                                  Shape must be `(batch_size, d_1, ..., d_k)`.
                                  
        Returns:
            torch.Tensor: The Focal-Dice loss as a scalar tensor.
        '''
        focal_loss = self.focal(logits, targs)
        dice_loss = self.dice(logits, targs)
        return (self.lambda_focal * focal_loss) + (self.lambda_dice * dice_loss)
    
    def forward_as_components(
        self, 
        logits: torch.Tensor, 
        targs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        '''
        Computes Focal-Dice loss and returns a dictionary containing
        each loss component and the total loss.

        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'focal' (torch.Tensor): The mean Focal loss component (unweighted).
                - 'dice' (torch.Tensor): The mean Dice loss component (unweighted).
                - 'total' (torch.Tensor): Weighted sum of mean Focal and mean Dice losses.
        '''
        focal_loss = self.focal(logits, targs)
        dice_loss = self.dice(logits, targs)
        tot_loss = (self.lambda_focal * focal_loss) + (self.lambda_dice * dice_loss)

        return {
            'focal': focal_loss,
            'dice': dice_loss,
            'total': tot_loss
        }
