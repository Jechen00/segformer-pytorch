#####################################
# Imports & Dependencies
#####################################
from __future__ import annotations

import torch

import numpy as np
from numbers import Real
from typing import Dict, Union, TypeAlias, Tuple, Optional, TypedDict, List

from src.ml_types import MeasureValue, MetricResults

#####################################
# Types
#####################################
HistoryValue: TypeAlias = Union[float, torch.Tensor]
HistoryGroup: TypeAlias = Dict[str, HistoryValue]
HistoryResults: TypeAlias = Dict[str, Union[HistoryValue, HistoryGroup]]

HistorySeries: TypeAlias = List[HistoryValue]
HistorySeriesGroup: TypeAlias = Dict[str, HistorySeries]

class PhaseHistoryState(TypedDict):
    loss: LossHistoryState
    metrics: Optional[MetricHistoryState]
    best_score: Optional[BestScoreDict]

class LossHistoryState(TypedDict):
    values: HistorySeries
    epochs: List[int]

class MetricHistoryState(TypedDict):
    values: Dict[str, Union[HistorySeries, HistorySeriesGroup]]
    epochs: List[int]

class BestScoreDict(TypedDict):
    value: float
    epoch: int
    measure_info: Optional[Union[str, Dict[str, str]]]


#####################################
# Functions
#####################################
def postprocess_measure(value: MeasureValue) -> HistoryValue:
    '''
    Postprocesses a measure value (e.g. loss or metric) into a format suitable for logging and saving.
    
    The procedure is as follows:
        - Detach tensors from computational graph (they should ideally be already detached).
        - Moves non-scalar (multi-element) tensors to CPU.
        - Converts non-scalar (multi-element) numpy arrays to tensors (on CPU).
        - Converts scalar (single-element) tensors/arrays to Python floats.
        - Converts real numeric inputs to Python floats.

    Args:
        value (MeasureValue): Computed measure value (a tensor, numpy array, or a real numeric value).

    Returns:
        HistoryValue: Processed measure value (a tensor or float) for logging and saving.
    '''
    if isinstance(value, torch.Tensor):
        value = value.detach()
        return float(value.item()) if value.numel() == 1 else value.cpu()
        
    elif isinstance(value, np.ndarray):
        return float(value.item()) if value.size == 1 else torch.from_numpy(value)
        
    elif isinstance(value, Real):
        return float(value)
    
    else:
        raise TypeError(f'Expected type torch.Tensor, np.ndarray, or Real. Got {type(value)}.')


#####################################
# Measure Histories
#####################################
class LossHistory():
    '''
    Stores loss values over training epochs.
    '''
    def __init__(self):
        self.values = []
        self.epochs = []
        
    def record(self, loss_value: MeasureValue, epoch: int) -> HistoryValue:
        '''
        Stores the loss value for a specified epoch.
        '''
        self.epochs.append(epoch)

        recorded_value = postprocess_measure(loss_value)
        self.values.append(recorded_value)

        return recorded_value

    def state_dict(self) -> LossHistoryState:
        return {
            'values': self.values,
            'epochs': self.epochs
        }
    
    def load_state_dict(self, state_dict: LossHistoryState) -> None:
        self.values = state_dict['values']
        self.epochs = state_dict['epochs']
        

class MetricHistory():
    '''
    Stores evaluation metric values over training epochs.
    '''
    def __init__(self):
        self.values = {}
        self.epochs = []
        
    def record(self, metric_results: MetricResults, epoch: int) -> HistoryResults:
        '''
        Stores the evaluation metric values for a specified epoch.
        '''
        self.epochs.append(epoch)
        
        recorded_metrics = {}
        for result_name, metric_result in metric_results.items():
            if isinstance(metric_result, dict):
                # metric_result is a dictionary of metric values
                recorded_metrics[result_name] = {}
                result_values = self.values.setdefault(result_name, {})
                for metric_name, metric_value in metric_result.items():
                    recorded_value = postprocess_measure(metric_value)
                    recorded_metrics[result_name][metric_name] = recorded_value

                    metric_values = result_values.setdefault(metric_name, [])
                    metric_values.append(recorded_value)
                    
            else:
                # metric_result is a single metric value
                recorded_value = postprocess_measure(metric_result)
                recorded_metrics[result_name] = recorded_value

                metric_values = self.values.setdefault(result_name, [])
                metric_values.append(recorded_value)

        return recorded_metrics

    def state_dict(self) -> MetricHistoryState:
        return {
            'values': self.values,
            'epochs': self.epochs
        }
    
    def load_state_dict(self, state_dict: MetricHistoryState) -> None:
        self.values = state_dict['values']
        self.epochs = state_dict['epochs']


#####################################
# Phase Histories
#####################################
class PhaseHistory():
    '''
    Stores the histories (e.g. loss and metrics) for a phase (e.g. training or validation).
    '''
    def __init__(self, loss: Optional[LossHistory] = None, metrics: Optional[MetricHistory] = None):
        self.loss = LossHistory() if loss is None else loss
        self.metrics = metrics
        self.best_score = None

    def record(
        self, 
        loss_value: MeasureValue, 
        epoch: int, 
        metric_results: Optional[MetricResults] = None
    ) -> Tuple[HistoryValue, Optional[HistoryResults]]:
        recorded_loss = self.loss.record(loss_value, epoch)

        if metric_results is None:
            return recorded_loss, None
        
        if self.metrics is None:
            raise ValueError(
                'metric_results were provided, but this PhaseHistory '
                'was initialized without a MetricHistory (metrics = None).'
            )
        
        recorded_metrics = self.metrics.record(metric_results, epoch)
        return recorded_loss, recorded_metrics
    
    def set_best(
        self, 
        value: float, 
        epoch: int, 
        measure_info: Optional[Union[str, Dict[str, str]]] = None
    ) -> None:
        self.best_score = {
            'value': value,
            'epoch': epoch,
            'measure_info': measure_info
        }

    def state_dict(self) -> PhaseHistoryState:
        return {
            'loss': self.loss.state_dict(),
            'metrics': None if self.metrics is None else self.metrics.state_dict(),
            'best_score': self.best_score
        }
    
    def load_state_dict(self, state_dict: PhaseHistoryState) -> None:
        # Load loss history
        self.loss.load_state_dict(state_dict['loss'])

        # Load metric history
        metrics = self.metrics
        metrics_state_dict = state_dict.get('metrics', None)

        if (metrics is None) and (metrics_state_dict is not None):
            raise ValueError(
                "state_dict contains a 'metrics', but this PhaseHistory "
                'was initialized without a MetricHistory (metrics = None).'
            )
        elif (metrics is not None) and (metrics_state_dict is None):
            raise KeyError(
                'This PhaseHistory was initialzied with a MetricHistory, '
                "but state_dict is missing 'metrics'."
            )
        
        if metrics is not None:
            metrics.load_state_dict(metrics_state_dict)

        # Load best metric score
        self.best_score = state_dict.get('best_score', None)


class TrainHistory(PhaseHistory):
    '''
    Training-phase history.
    This only tracks loss information.
    '''
    def __init__(self):
        super().__init__(loss = LossHistory())


class ValHistory(PhaseHistory):
    '''
    Validation-phase history.
    This tracks both loss and metric information.
    '''
    def __init__(self):
        super().__init__(loss = LossHistory(), metrics = MetricHistory())