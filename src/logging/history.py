#####################################
# Imports & Dependencies
#####################################
from __future__ import annotations

from typing import Dict, Any, Tuple, Optional, TypedDict, List

from src.metrics.postprocess import format_measure
from src.metrics.types import (
    MeasureValue,  MetricResults, MeasureSeries, MetricSeriesResults
)



#####################################
# Typed Dicts
#####################################
class PhaseHistoryState(TypedDict):
    loss: LossHistoryState
    metrics: Optional[MetricHistoryState]
    best_score: Optional[BestScoreDict]

class LossHistoryState(TypedDict):
    values: MeasureSeries
    epochs: List[int]

class MetricHistoryState(TypedDict):
    values: MetricSeriesResults
    epochs: List[int]

class BestScoreDict(TypedDict):
    value: float
    epoch: int
    measure_info: Optional[Dict[str, Any]]


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
        
    def record(self, loss_value: MeasureValue, epoch: int) -> MeasureValue:
        '''
        Formats and stores the loss value for a specified epoch.

        Note: See `src.metrics.postprocess.format_measure` for details on formatting.

        Args:
            loss_value (MeasureValue): 
                Loss value in the form of a float or a single-element tensor.
            epoch (int): 
                The epoch that `loss_value` corresponds to.

        Returns:
            MeasureValue: 
                The loss value, formatted for logging and saving.             
        '''
        self.epochs.append(epoch)

        recorded_value = format_measure(loss_value)
        self.values.append(recorded_value)

        return recorded_value

    def state_dict(self) -> LossHistoryState:
        '''
        Returns the state dictionary for this `LossHistory` instance.

        Returns:
            LossHistoryState: 
                State dictionary containing:
                    - values (MeasureSeries): List of loss values recorded across epochs.
                    - epochs (List[int]): List of epochs corresponding to recorded losses in `values`.
        '''
        return {
            'values': self.values,
            'epochs': self.epochs
        }
    
    def load_state_dict(self, state_dict: LossHistoryState) -> None:
        '''
        Loads a state dictionary into this `LossHistory` instance.

        Args:
            state_dict (LossHistoryState): 
                State dictionary to load.
                This must contain:
                    - values (MeasureSeries): List of loss values recorded across epochs.
                    - epochs (List[int]): List of epochs corresponding to recorded losses in `values`.
        '''
        self.values = state_dict['values']
        self.epochs = state_dict['epochs']
        

class MetricHistory():
    '''
    Stores evaluation metric values over training epochs.
    '''
    def __init__(self):
        self.values = {}
        self.epochs = []
        
    def record(self, metric_results: MetricResults, epoch: int) -> MetricResults:
        '''
        Formats and stores the values from a metric result dictionary (`MetricResults`) for a specified epoch.

        Note: See `src.metrics.postprocess.format_measure` for details on formatting
        individual metric values (leaf values in the result dictionary).

        Note: See `src.metrics.types` for an example structure of `MetricResults`.

        Args:
            metric_results (MetricResults): 
                Metric result dictionary to format and store.
            epoch (int): 
                The epoch corresponding to `metric_results`.

        Returns:
            MetricResults: 
                The metric result dictionary, with all of its values formatted for logging and saving.
                The layout is the same as the input `metric_results`.
        '''
        self.epochs.append(epoch)
        
        recorded_metrics = {}
        for result_name, metric_result in metric_results.items():
            if isinstance(metric_result, dict):
                # metric_result is a dictionary of metric values
                recorded_metrics[result_name] = {}
                result_values = self.values.setdefault(result_name, {})
                for metric_name, metric_value in metric_result.items():
                    recorded_value = format_measure(metric_value)
                    recorded_metrics[result_name][metric_name] = recorded_value

                    metric_values = result_values.setdefault(metric_name, [])
                    metric_values.append(recorded_value)
                    
            else:
                # metric_result is a single metric value
                recorded_value = format_measure(metric_result)
                recorded_metrics[result_name] = recorded_value

                metric_values = self.values.setdefault(result_name, [])
                metric_values.append(recorded_value)

        return recorded_metrics

    def state_dict(self) -> MetricHistoryState:
        '''
        Returns the state dictionary for this `MetricHistory` instance.

        Returns:
            MetricHistoryState: 
                State dictionary containing:
                    - values (MetricSeriesResults): Dictionary containing metric result lists.
                                                    See `src.metrics.types` for 
                                                    an example structure of `MetricSeriesResults`.
                    - epochs (List[int]): List of epochs corresponding to recorded values in `values`.
        '''
        return {
            'values': self.values,
            'epochs': self.epochs
        }

    def load_state_dict(self, state_dict: MetricHistoryState) -> None:
        '''
        Loads a state dictionary into this `MetricHistory` instance.

        Args:
            state_dict (MetricHistoryState): 
                State dictionary to load.
                This must contain:
                    - values (MetricSeriesResults): Dictionary containing metric result lists.
                                                    See `src.metrics.types` for 
                                                    an example structure of `MetricSeriesResults`.
                    - epochs (List[int]): List of epochs corresponding to recorded values in `values`.
        '''
        self.values = state_dict['values']
        self.epochs = state_dict['epochs']


#####################################
# Phase Histories
#####################################
class PhaseHistory():
    '''
    Stores the histories (e.g. loss and metrics) for a phase (e.g. training or validation).

    Important attributes:
        - `loss.values` (MeasureSeries): Series of recorded loss values.
        - `loss.epochs` (List[int]): Series of epochs corresponding to `self.loss.values`.

        - `metrics.values` (MetricSeriesGroup): Dictionary containing series of recorded metric values.
        - `metrics.epochs` (List[int]): Series of epochs corresponding to `metrics.values`.
        
    Args:
        loss (optional, LossHistory): 
            Loss history object for tracking loss values over epochs.
            If `None`, defaults to a fresh `LossHistory` instance.
        metrics (optional, MetricHistory): 
            Metric history object for tracking metric values over epochs.
            If `None`, no metrics are recorded.
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
    ) -> Tuple[MeasureValue, Optional[MetricResults]]:
        '''
        Formats and stores loss and optional metric values for a specified epoch.

        Note: Optional metric values are from a metric result dictionary (`MetricResults`),
        of which an example structure an be found in `src.metrics.types`.

        Note: Metric values can only be stored if a metric history 
              was provided at initialization (`metrics` is not `None`).
              If no metric history was provided, a `ValueError` will be raised
              when attempting to store a metric result dictionary.

        Args:
            loss_value (MeasureValue): 
                Loss value in the form of a float or a single-element tensor.
            epoch (int): 
                The epoch corresponding to `loss_value` and `metric_results`.
            metric_results (optional, MetricResults): 
                Metric result dictionary to format and store.

        Returns:
            recorded_loss (MeasureValue): 
                The loss value, formatted for logging and saving.   
            recorded_metrics (optional, MetricResults): 
                The metric result dictionary, with all of its values formatted for logging and saving.  
                The layout is the same as the input `metric_results`.
                If `metric_results` was not provided, this is returned as `None`.

        '''
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
        measure_info: Optional[Dict[str, Any]] = None
    ) -> None:
        '''
        Sets a best score for the phase history.
        This is stored in the attribute `self.best_score` as a dictionary:
            {
                'value': value,
                'epoch': epoch,
                'measure_info': measure_info
            }

        Args:
            value (float): 
                The best score value.
            epoch (int): 
                The epoch corresponding to `value`.
            measure_info (optional, Dict[str, Any]): 
                Dictionary containing additional information for the best score.
                This could include information like the measure name (e.g. accuracy).
        '''
        self.best_score = {
            'value': value,
            'epoch': epoch,
            'measure_info': measure_info
        }

    def state_dict(self) -> PhaseHistoryState:
        '''
        Returns the state dictionary for this `PhaseHistory` instance.

        Returns:
            PhaseHistoryState: 
                State dictionary containing:
                    - loss (LossHistoryState): State dictionary for the loss history.
                    - metrics (optional, MetricHistoryState): State dictionary for the metric history.
                                                              This is `None` if no metric history was
                                                              provided at initialization (`metrics` is None).
                    - best_score (optional, BestScoreDict): Best score dictionary stored in this phase history.
                                                            See the `set_best` method for details on its contents.
                                                            This is `None` if a best score was never set 
                                                            (`set_best` was never called.).
        '''
        return {
            'loss': self.loss.state_dict(),
            'metrics': None if self.metrics is None else self.metrics.state_dict(),
            'best_score': self.best_score
        }
    
    def load_state_dict(self, state_dict: PhaseHistoryState) -> None:
        '''
        Loads a state dictionary into this `PhaseHistory` instance.

        Note: To load a metric history state dictionary (`MetricHistoryState`),
              a metric history must be provided at initialization (`metrics` is not `None`).
              Otherwise, a `ValueError` will be raised when attempting to load a `MetricHistoryState`.

        Args:
            state_dict (PhaseHistoryState): 
                State dictionary to load.
                This must contain:
                    - loss (LossHistoryState): State dictionary for the loss history.
                    - metrics (optional, MetricHistoryState): State dictionary for the metric history,
                                                              if one was provided at initialization.
                    - best_score (optional, BestScoreDict): Best score dictionary.
                                                            See the `set_best` method for the expected structure.
        '''
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
    
    Important attributes:
        - `loss.values` (MeasureSeries): Series of recorded loss values.
        - `loss.epochs` (List[int]): Series of epochs corresponding to `self.loss.values`.
    '''
    def __init__(self):
        super().__init__(loss = LossHistory())


class ValHistory(PhaseHistory):
    '''
    Validation-phase history.
    This tracks both loss and metric information.

    Important attributes:
        - `loss.values` (MeasureSeries): Series of recorded loss values.
        - `loss.epochs` (List[int]): Series of epochs corresponding to `loss.values`.

        - `metrics.values` (MetricSeriesGroup): Dictionary containing series of recorded metric values.
        - `metrics.epochs` (List[int]): Series of epochs corresponding to `metrics.values`.
    '''
    def __init__(self):
        super().__init__(loss = LossHistory(), metrics = MetricHistory())