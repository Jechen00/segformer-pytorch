#####################################
# Imports & Dependencies
#####################################
import torch

import warnings
from typing import Optional, Literal, Tuple, TypedDict, NotRequired

from src.metrics.postprocess import (
    MetricSpecLike, format_metric_spec, select_and_agg_scalar_metric
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
    '''
    Creates a measure policy for defining and tracking a best score.
    Also optionally determines early stopping.

    Note: If `measure_type='metric'`, the metric specification must
          produce a scalar value after extracting and optionally subsetting and/or aggregating
          from the provided `MetricResults`.

    Args:
        measure_type (Literal['loss', 'metric']): 
            Specifies whether the best score is defined as a loss or as a metric.
        metric_spec (optional, MetricSpecLike): 
            Specification used to extract and optionally subset and/or aggregate a metric for the best score.
            Required when `measure_type='metric'`.
            Supported types:
                - `MetricSpec` instance.
                - `MetricSpecDict` dictionary.
        mode (Literal['min', 'max']): 
            The rule used to determine the best score.
                - 'min': The best score is the minimum value of the defined measure.
                - 'max': The best score is the maximum value of the defined measure.
        min_delta (float): 
            The minimum deviation from the current best score 
            for an update/step to be considered an improvement.
            Default is `0.0`.
        patience (optional, int): 
            The maximum number of consecutive updates/steps without improvement
            before triggering early stopping (`should_stop=True`).
            Default is `None`, meaning that early stopping is disabled.
    '''
    def __init__(
        self,
        measure_type: Literal['loss', 'metric'],
        mode: Literal['min', 'max'],
        metric_spec: Optional[MetricSpecLike] = None,
        min_delta: float = 0.0,
        patience: Optional[int] = None
    ):
        self.measure_type = measure_type
        self.metric_spec = format_metric_spec(metric_spec) if metric_spec is not None else None
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self._validate_configs()

        self.measure_info = self._make_measure_info()
        self.no_improve_counter = 0
        self.best_score = None
        
    def step(self, measure: MeasureResult) -> Tuple[bool, bool]:
        '''
        Updates the measure policy.
        This involves:
            1. Extracting a scalar score from `measure`.
            2. Comparing extracted score with `best_score`.
                - If the score is an improvement, update `best_score`.
            3. Optionally checking for early stopping (if `patience` is not `None`)

        Note: See `src.metrics.types` for example structures of `MetricGroup` and `MetricResults`.

        Args:
            measure (MeasureResult): 
                A loss value or metric dictionary, depending on `measure_type`.
                    - `measure_type = 'loss'` : Loss value (`MeasureValue`) as a float or tensor.
                    - `measure_type = 'metric'`: Metric dictionary (`MetricGroup` or `MetricResults`), 
                                                 where `metric_spec` will be used to extract a metric
                                                 and optionally subset and/or aggregate it.

        Returns:
            improved (bool): 
                Whether the extracted score is an improvement over `best_score`.
            should_stop (bool): 
                Whether early stopping has been triggered.
                If `patience` is `None` at initalization, this is always `False`.
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
        '''
        Extracts a scalar score to compare with `best_score`.

        Note: See `src.metrics.types` for example structures of `MetricGroup` and `MetricResults`.

        Args:
            measure (MeasureResult): 
                A loss value or metric dictionary, depending on `measure_type`.
                    - `measure_type = 'loss'`: Loss value (`MeasureValue`) as a float or tensor.
                    - `measure_type = 'metric'`: Metric dictionary (`MetricGroup` or `MetricResults`), 
                                                 where `metric_spec` will be used to extract a metric
                                                 and optionally subset and/or aggregate it.
                                                
        Returns:
            float: 
                The extracted scalar score.
        '''
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
        Returns the state dictionary for this `MeasurePolicy` instance.

        Returns:
            MeasurePolicyState: 
                State dictionary containing:
                    - 'no_improve_counter' (int): Integer counting the number of consecutive 
                                                updates/steps without improvement in `best_score`.
                    - 'best_score' (optional, float): The current best score for the defined measure.
                                                    If there has not been a single update/step,
                                                    this will be `None`.
                                                
        '''
        return {
            'no_improve_counter': self.no_improve_counter,
            'best_score': self.best_score
        }
    
    def load_state_dict(self, state_dict: MeasurePolicyState) -> None:
        '''
        Loads a state dictionary into this `MeasurePolicy` instance.

        Args:
            MeasurePolicyState: 
                State dictionary containing:
                    - 'no_improve_counter' (int): Integer counting the number of consecutive 
                                                updates/steps without improvement in `best_score`.
                    - 'best_score' (optional, float): The current best score for the defined measure.
        '''
        self.no_improve_counter = state_dict['no_improve_counter']
        self.best_score = state_dict['best_score']

    def _validate_configs(self) -> None:
        '''
        Validates the arguments for initializing the `MeasurePolicy` instance.
        '''
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
        '''
        Creates a dictionary containing information on the defined measure.

        Returns:
            MeasureInfo: 
                Measure information dictionary.
                    - If `measure_type='loss'`: 
                        Contains only the measure type.
                    - If `measure_type='metric'`: 
                        Contains the measure type, and all attributes from `metric_spec`.
        '''
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