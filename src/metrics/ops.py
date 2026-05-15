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


class ClassificationMetrics():
    '''
    Computes accuracy, precision, recall, and F1 score.
    Supports sample-wise and element-wise classification:
        - Sample-wise (e.g. image classification): metrics are computed over batch samples.
        - Element-wise (e.g. segmentation): metrics are computed over all elements across the batch.

    Args:
        num_classes (int): Number of classes in the dataset.
        ignore_idx (optional, int): Index for an 'ignore class'.
                                    Targets with this index will be ignored (along with their predictions)
                                    when computing the confusion matrix and related metrics.
    '''
    def __init__(self, num_classes: int, ignore_idx: Optional[int] = None):
        self.num_classes = num_classes
        self.conf_mat = None
        self.ignore_idx = ignore_idx
        
    def update(self, preds: torch.Tensor, targs: torch.Tensor) -> None:
        '''
        Updates confusion matrix between predictions and targets.

        Args:
            preds (torch.Tensor): Class prediction tensor.
                                  Shape depends on the task:
                                        - Sample-wise: `(batch_size,)`
                                        - Element-wise: `(batch_size, height, width)`
            targs (torch.Tensor): Target tensor containing ground truth class indices.
                                  Shape depends on the task:
                                        - Sample-wise: `(batch_size,)`
                                        - Element-wise: `(batch_size, height, width)`
        '''
        preds = preds.flatten()
        targs = targs.flatten()
        
        if self.ignore_idx is not None:
            valid_mask = (targs != self.ignore_idx)
            preds = preds[valid_mask]
            targs = targs[valid_mask]

        present_classes = torch.concat([preds, targs])
        max_class = present_classes.max().item()
        min_class = present_classes.min().item()

        if (min_class < 0) or (max_class >= self.num_classes):
            raise ValueError(
                f'Class indices should be in [0, {self.num_classes - 1}], '
                f'but got range [{min_class}, {max_class}].'
            )
        
        update_conf_mat = torch.bincount(
            self.num_classes * targs + preds,
            minlength = self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
            
        if self.conf_mat is not None:
            self.conf_mat += update_conf_mat
        else:
            self.conf_mat = update_conf_mat
            
    def compute(self) -> MetricGroup:
        '''
        Computes all classification metrics across all updated target elements.

        Returns:
            MetricGroup: Metric dictionary containing
                - 'accuracy (int): Total accuracy computed across all classes.
                - precision (torch.Tensor): Per class precision tensor of shape `(num_classes,)`.
                - recall (torch.Tensor): Per class recall tensor of shape `(num_classes,)`.
                - f1_score (torch.Tensor): Per class F1-score tensor of shape `(num_classes,)`.
                - tot_true_pos (int): Total number of true positives (correct predictions).
                - tot_count (int): Total number of target elements.
        '''
        if self.conf_mat is None:
            raise RuntimeError('Metric has not been updated with any data yet.')
            
        true_pos = self.conf_mat.diag() # True positives (correct predictions) per class
        class_count = self.conf_mat.sum(dim = 1) # Count of target elements per class
        pos_count = self.conf_mat.sum(dim = 0) # Positive classifications per class

        tot_count = class_count.sum() # Total count of target elements
        tot_true_pos =  true_pos.sum() # Total true positives (correct predictions)

        accuracy = (tot_true_pos / tot_count) if tot_count > 0 else 0.0
        recall = torch.where(class_count > 0, true_pos / class_count, 0.0)
        precision = torch.where(pos_count > 0, true_pos / pos_count, 0.0)
        f1_score = torch.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0
        )

        return {
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1_score': f1_score,
            'tot_true_pos': tot_true_pos,
            'tot_count': tot_count
        }
    
    def reset(self) -> None:
        '''
        Resets confusion matrix.
        '''
        self.conf_mat = None