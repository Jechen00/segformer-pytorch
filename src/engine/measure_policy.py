#####################################
# Imports & Dependencies
#####################################
import torch

import numpy as np
import warnings
from typing import Optional, Dict, Literal, Union, Tuple

from src.ml_types import MeasureValue, MetricGroup
from src.utils import nested_extract, apply_agg


#####################################
# Classes
#####################################
class MeasurePolicy():
    '''
    metric_path (Optional[str]): Dot-separated key path specifying the key sequence used to extract 
                                 a numeric value, tensor, or numpy array from a pontially nested metric dictionary.
                                 Required if `measure_type = 'metric'` and ignored otherwise.
    '''
    def __init__(
        self,
        measure_type: Literal['loss', 'metric'],
        metric_path: Optional[str] = None,
        metric_agg: Literal['mean', 'min', 'max'] = 'mean',
        mode: Literal['min', 'max'] = 'max',
        min_delta: float = 0.0,
        patience: Optional[int] = None
    ):
        self.measure_type = measure_type
        self.metric_path = metric_path
        self.metric_agg = metric_agg
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self._validate_configs()

        if measure_type == 'loss':
            if metric_path is not None:
                warnings.warn(
                    "metric_path is ignored when measure_type == 'loss'.",
                    UserWarning
                )
            self.measure_info = {'measure_type': 'loss'}
        else:
            if metric_path is None:
                raise ValueError(
                    "metric_path must be provided when measure_type == 'metric'."
                )
            self.measure_info = {
                'measure_type': 'metric',
                'metric_path': metric_path,
                'metric_agg': metric_agg
            }
        
        self.no_improve_counter = 0
        self.best_score = None
        
    def step(self, measure: Union[MeasureValue, MetricGroup]) -> Tuple[bool, bool]:
        '''
        Args:
            measure (Union[float, MetricGroup]): A loss value or metric dictionary, depending on `self.measure_type`.
                - `measure_type = 'loss'`: Loss value represented by a numeric value, tenosr, or numpy array.
                - `measure_type = 'metric'`: Metric dictionary, where `self.metric_path` leads to
                                            a numeric value, tensor, or numpy array.
                                            If tensor or numpy array, it is aggregated using `self.metric_agg`.
        '''
        score = self.extract_score(measure)

        # Check for improvement
        if self.best_score is None:
            improved = True
        elif self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        elif self.mode == 'min':
            improved = score < self.best_score - self.min_delta

        # Determine early stopping and update best score if needed
        if improved:
            self.no_improve_counter = 0
            self.best_score = score
        else:
            self.no_improve_counter += 1

        should_stop = (
            (self.patience is not None)
            and (self.no_improve_counter >= self.patience)
        )

        return improved, should_stop

    def extract_score(self, measure: Union[MeasureValue, MetricGroup]) -> float:
        if self.measure_type == 'loss':
            score = measure
        else:
            # Extract from nested metric dictionary
            score = nested_extract(measure, self.metric_path)

        # Aggregate if needed
        if isinstance(score, (torch.Tensor, np.ndarray)):
            # Should work on single-element tensors/arrays
            score = apply_agg(score, self.metric_agg) # This always returns a float

        return float(score)

    def reset(self):
        '''
        Resets state information.
        '''
        self.no_improve_counter = 0
        self.best_score = None

    def state_dict(self) -> Dict[str, float]:
        '''
        Returns state information as a dictionary.
        '''
        return {
            'no_improve_counter': self.no_improve_counter,
            'best_score': self.best_score
        }
    
    def load_state_dict(self, state_dict):
        '''
        Loads state information from a state_dict.
        '''
        self.no_improve_counter = state_dict['no_improve_counter']
        self.best_score = state_dict['best_score']

    def _validate_configs(self):
        if self.measure_type not in ['loss', 'metric']:
            raise ValueError("measure_type must be 'loss' or 'metric'.")

        if self.metric_agg not in ['mean', 'max', 'min']:
            raise ValueError("metric_agg must be 'mean', 'min', or 'max'.")
            
        if self.mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'.")
        
        if self.min_delta < 0:
            raise ValueError('min_delta must be non-negative.')
        
        if (self.patience is not None) and (self.patience < 1):
            raise ValueError('patience must be greater than 0.')
        