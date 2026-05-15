#####################################
# Imports & Dependencies
#####################################
import torch

import warnings
from typing import Optional, Literal, Tuple, TypedDict, NotRequired

from src.metrics.postprocess import (
    MetricSpecLike, normalize_metric_spec, select_and_agg_scalar_metric
)
from src.metrics.types import MeasureResult
from src.ml_types import IndexLike, Aggregation


#####################################
# State Dict
#####################################
class MeasurePolicyState(TypedDict):
    no_improve_counter: int
    best_score: Optional[float]

class MeasureInfo(TypedDict):
    measure_type: str
    key_path: NotRequired[Optional[str]]
    class_idxs: NotRequired[Optional[IndexLike]]
    agg: NotRequired[Optional[Aggregation]]
    unit: NotRequired[Optional[str]]


#####################################
# Measure Policy Class
#####################################
class MeasurePolicy():
    def __init__(
        self,
        measure_type: Literal['loss', 'metric'],
        metric_spec: Optional[MetricSpecLike] = None,
        mode: Literal['min', 'max'] = 'max',
        min_delta: float = 0.0,
        patience: Optional[int] = None
    ):
        self.measure_type = measure_type
        self.metric_spec = normalize_metric_spec(metric_spec) if metric_spec is not None else None
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self._validate_configs()

        self.measure_info = self._make_measure_info()
        self.no_improve_counter = 0
        self.best_score = None
        
    def step(self, measure: MeasureResult) -> Tuple[bool, bool]:
        '''
        Args:
            measure (MeasureResult): 
                A loss value or metric dictionary, depending on `self.measure_type`.
                    - `measure_type = 'loss'` : Loss value (MeasureValue) as a float or tensor.
                    - `measure_type = 'metric'`: Metric dictionary (Union[MetricGroup, MetricResults]), 
                                                 where `self.metric_spec` will be used to extract a metric
                                                 and optionally subset and aggregate it.
                                                 It is expected that `self.metric_spec` produces
                                                 a float or a single-element tensor.
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

    def extract_score(self, measure: MeasureResult) -> float:
        if self.measure_type == 'loss':
            score = measure.item() if isinstance(measure, torch.Tensor) else measure

        else:
            # Select and aggregate from nested metric dictionary
            metric_spec = self.metric_spec
            score = select_and_agg_scalar_metric(
                metric_data = measure,
                key_path = metric_spec.key_path,
                class_idxs = metric_spec.class_idxs,
                agg = metric_spec.agg
            )
        
        return float(score)

    def reset(self) -> None:
        '''
        Resets state information.
        '''
        self.no_improve_counter = 0
        self.best_score = None

    def state_dict(self) -> MeasurePolicyState:
        '''
        Returns state information as a dictionary.
        '''
        return {
            'no_improve_counter': self.no_improve_counter,
            'best_score': self.best_score
        }
    
    def load_state_dict(self, state_dict: MeasurePolicyState) -> None:
        '''
        Loads state information from a state_dict.
        '''
        self.no_improve_counter = state_dict['no_improve_counter']
        self.best_score = state_dict['best_score']

    def _validate_configs(self) -> None:
        if self.measure_type == 'loss':
            if self.metric_spec is not None:
                warnings.warn(
                    "metric_spec is ignored when measure_type='loss'.",
                    UserWarning
                )
        elif self.measure_type == 'metric':
            if self.metric_spec is None:
                raise ValueError(
                    "metric_spec must be provided when measure_type='metric'."
                )
        else:
            raise ValueError("measure_type must be 'loss' or 'metric'.")

        if self.mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'.")
        
        if self.min_delta < 0:
            raise ValueError('min_delta must be non-negative.')
        
        if (self.patience is not None) and (self.patience < 1):
            raise ValueError('patience must be greater than 0.')
        
    def _make_measure_info(self) -> MeasureInfo:
        if self.measure_type == 'loss':
            return {'measure_type': 'loss'}
        else:
            metric_spec = self.metric_spec
            return {
                'measure_type': 'metric',
                'key_path': metric_spec.key_path,
                'class_idxs': metric_spec.class_idxs,
                'agg': metric_spec.agg,
                'unit': metric_spec.unit
            }