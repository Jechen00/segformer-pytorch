#####################################
# Imports & Dependencies
#####################################
import torch

from typing import Protocol, Union, Optional

from src.metrics.types import MeasureValue, MetricGroup


#####################################
# Metric Classes
#####################################
class Metric(Protocol):
    '''
    Protocol for evaluation metrics.
    '''
    def update(self, preds: torch.Tensor, targs: torch.Tensor) -> None: ...
    def compute(self) -> Union[MeasureValue, MetricGroup]: ...
    def reset(self) -> None: ...


class ConfusionMatrix():
    '''
    Accumulates a confusion matrix between predictions and targets 
    for single-label classification tasks.

    Inputs must be integer class indices in the range `[0, num_classes - 1]`.
    The only exception is the `ignore_idx` (if specified).

    Args:
        num_classes (int): 
            Number of classes in the dataset.
        ignore_idx (optional, int): 
            Index to ignore during computation.
            Targets with this index (and their predictions) are ignored
            when computing the confusion matrix and related metrics.
    '''
    def __init__(self, num_classes: int, ignore_idx: Optional[int] = None):
        self.num_classes = num_classes
        self.ignore_idx = ignore_idx
        self.conf_mat = None # Row: target index, Col: predicted index

    def update(self, preds: torch.Tensor, targs: torch.Tensor) -> None:
        '''
        Updates confusion matrix between predictions and targets.

        Args:
            preds (torch.Tensor): 
                Prediction tensor for class indices.
                Must consist of integer class indices.
            targs (torch.Tensor): 
                Target (ground truth) tensor for class indices.
                Must be the same shape as `preds` 
                and consist of integer class indices.
        '''
        if preds.shape != targs.shape:
            raise ValueError(
                'preds and targs must be the same shape. '
                f'Got: preds.shape={preds.shape}, targs.shape={targs.shape} '
            )

        preds = preds.flatten()
        targs = targs.flatten()
        
        if self.ignore_idx is not None:
            valid_mask = (targs != self.ignore_idx)
            preds = preds[valid_mask]
            targs = targs[valid_mask]

            if targs.numel() == 0:
                return

        min_class = min(preds.min(), targs.min()).item()
        max_class = max(preds.max(), targs.max()).item()

        num_classes = self.num_classes
        if (min_class < 0) or (max_class >= num_classes):
            raise ValueError(
                f'Class indices should be in [0, {num_classes - 1}], '
                f'but got range [{min_class}, {max_class}].'
            )
        
        update_conf_mat = torch.bincount(
            num_classes * targs + preds,
            minlength = num_classes**2
        ).reshape(num_classes, num_classes).to(dtype = torch.float32)
            
        if self.conf_mat is not None:
            self.conf_mat += update_conf_mat
        else:
            self.conf_mat = update_conf_mat

    def compute(self) -> torch.Tensor:
        '''
        Returns confusion matrix.
        '''
        return self.conf_mat
    
    def reset(self) -> None:
        '''
        Resets confusion matrix.
        '''
        self.conf_mat = None


class ClassificationMetrics(ConfusionMatrix):
    '''
    Evaluation metrics for single-label classification, computed from a confusion matrix.
    Includes: accuracy, precision, recall, F1-score

    Inputs must be integer class indices in the range `[0, num_classes - 1]`.
    The only exception is the `ignore_idx` (if specified).

    Args:
        num_classes (int): 
            Number of classes in the dataset.
        ignore_idx (optional, int): 
            Index to ignore during computation.
            Targets with this index (and their predictions) are ignored
            when computing the confusion matrix and related metrics.
    '''     
    def compute(self) -> MetricGroup:
        '''
        Computes the classification metrics across all target-prediction updates.

        Returns:
            MetricGroup: Metric dictionary containing
                - accuracy (torch.Tensor): Overall accuracy across all classes. 
                                           This is a scalar tensor.
                - recall (torch.Tensor): Per-class recall tensor of shape `(num_classes,)`.
                - mean_recall (torch.Tensor): Mean recall across all classes 
                                              with at least one target or predicted element.
                                              This is a scalar tensor.
                - precision (torch.Tensor): Per-class precision tensor of shape `(num_classes,)`.
                - mean_precision (torch.Tensor): Mean precision across all classes 
                                                 with at least one target or predicted element.
                                                 This is a scalar tensor.
                - f1_score (torch.Tensor): Per-class F1-score tensor of shape `(num_classes,)`.
                - mean_f1_score (torch.Tensor): Mean F1-score across all classes 
                                                with at least one target or predicted element.
                                                This is a scalar tensor.
                - tot_count (torch.Tensor): Total number of non-ignored target elements.
                                            This is a scalar tensor.
        '''
        conf_mat = self.conf_mat
        if conf_mat is None:
            raise RuntimeError('Metric has not been updated with any data yet.')
            
        true_pos = conf_mat.diag() # True positives (correct predictions) per class
        class_count = conf_mat.sum(dim = 1) # Count of target elements per class
        pos_count = conf_mat.sum(dim = 0) # Count of prediction elements per class

        tot_count = class_count.sum() # Total count of elements considered (non-ignored targets)
        tot_true_pos =  true_pos.sum() # Total true positives (correct predictions)

        # Compute accuracy
        accuracy = (tot_true_pos / tot_count) if tot_count > 0 else 0.0

        # Compute recall
        recall = torch.zeros_like(true_pos)
        class_mask = class_count > 0 # Compute only where target elements exist for the class
        recall[class_mask] = true_pos[class_mask] / class_count[class_mask]

        # Compute precision
        precision = torch.zeros_like(true_pos)
        pos_mask = pos_count > 0 # Compute only where prediction elements exist for the class
        precision[pos_mask] = true_pos[pos_mask] / pos_count[pos_mask]

        # Comput F1-score
        f1_score = torch.zeros_like(true_pos)
        f1_denom = precision + recall
        f1_mask = f1_denom > 0 # Compute only where denominator is nonzero
        f1_score[f1_mask] = (2 * precision[f1_mask] * recall[f1_mask]) / f1_denom[f1_mask]

        present_mask = class_mask | pos_mask # Mask for classes with at least one target or predicted element

        return {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score,
            'mean_recall': recall[present_mask].mean(),
            'mean_precision': precision[present_mask].mean(),
            'mean_f1_score': f1_score[present_mask].mean(),
            'tot_count': tot_count
        }


class SegmentationMetrics(ConfusionMatrix):
    '''
    Evaluation metrics frequently used in semantic segmentation, computed from a confusion matrix.
    Includes: accuracy, Dice (F1-score), IoU (Jaccard index)

    Inputs must be integer class indices in the range `[0, num_classes - 1]`.
    The only exception is the `ignore_idx` (if specified).

    Args:
        num_classes (int): 
            Number of classes in the dataset.
        ignore_idx (optional, int): 
            Index to ignore during computation.
            Targets with this index (and their predictions) are ignored
            when computing the confusion matrix and related metrics.
    '''
    def compute(self) -> MetricGroup:
        '''
        Computes the semantic segmentation metrics across all target-prediction updates.

        Returns:
            MetricGroup: Metric dictionary containing
                - accuracy (torch.Tensor): Overall accuracy across all classes. 
                                           This is a scalar tensor.
                - dice (torch.Tensor): Per-class Dice tensor of shape `(num_classes,)`.
                - mean_dice (torch.Tensor): Mean Dice across all classes 
                                            with at least one target or predicted element.
                                            This is a scalar tensor.
                - iou (torch.Tensor): Per-class IoU tensor of shape `(num_classes,)`.
                - mean_iou (torch.Tensor): Mean IoU across all classes 
                                          with at least one target or predicted element.
                                           This is a scalar tensor.                  
                - tot_count (torch.Tensor): Total number of non-ignored target elements.
                                            This is a scalar tensor.
        '''
        conf_mat = self.conf_mat
        if conf_mat is None:
            raise RuntimeError('Metric has not been updated with any data yet.')
            
        # Class-wise intersection between target and prediction elements
        # These are the true positives
        intersection = conf_mat.diag() # |T_c intersect P_c|

        # Class-wise union between target and prediction elements
        # From inclusion-exclusion principle: |T_c union P_c| = |T_c| + |P_c| - |T_c intersect P_c|
        union = conf_mat.sum(dim = 0) + conf_mat.sum(dim = 1) - intersection

        # Total count of elements considered (non-ignored targets)
        tot_count = conf_mat.sum()
        
        # Compute IoU, Dice, and pixel-wise accuracy
        iou = torch.zeros_like(intersection)
        nonzero_mask = intersection > 0 # Compute only where intersect is nonzero
        iou[nonzero_mask] = intersection[nonzero_mask] / union[nonzero_mask]

        dice = 2 * iou / (1 + iou)

        accuracy = (intersection.sum() / tot_count) if tot_count > 0 else 0.0
        
        # Mask for classes with at least one target or predicted element
        present_mask = (conf_mat.sum(dim = 1) > 0) | (conf_mat.sum(dim = 0) > 0)

        return {
            'accuracy': accuracy,
            'dice': dice,
            'mean_dice': dice[present_mask].mean(),
            'iou': iou,
            'mean_iou': iou[present_mask].mean(),
            'tot_count': tot_count
        }