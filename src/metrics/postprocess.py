# Functions that convert MetricResults

#####################################
# Imports & Dependencies
#####################################
import torch

from dataclasses import dataclass
from typing import Union, Optional

from typing import List, TypedDict, NotRequired, TypeAlias, get_args
from src.utils import nested_extract, apply_agg, normalize_idxs

from src.ml_types import IndexLike, Aggregation
from src.metrics.types import MeasureValue, MetricResults, MeasureSeries, MetricSeriesResults


#####################################
# Metric Spec Classes/Types
#####################################
@dataclass
class MetricSpec():
    '''
    Specification for selecting and optionally aggregating a metric.
    This class stores the config for `select_and_agg_metric`,
    along with metadata for the metric (e.g. unit).
    
    Attributes:
        key_path (str): Dot-separated key path to extract a metric from
                        a metric results dictionary (`MetricResults` or `MetricSeriesResults`).
        class_idxs (optional, IndexLike): Optional class indices used to subset tensors in the extracted metric.
        agg (optional, Aggregation): Optional aggregation method applied to tensors after extracting and subsetting.
                                     If provided, must be one of: 'mean', 'median', 'min', 'max'.
        unit (optional, str): Unit associated with the metric. If not provided, it is assumed to be unitless.
    '''
    key_path: str
    class_idxs: Optional[IndexLike] = None
    agg: Optional[Aggregation] = None
    unit: Optional[str] = None

    def __post_init__(self):
        # Normalize class indices to an integer or tuple of integers
        if self.class_idxs is not None:
            self.class_idxs = normalize_idxs(self.class_idxs)

        # Validate aggregation
        valid_aggs = get_args(Aggregation)
        agg = self.agg
        if (agg is not None) and (agg not in valid_aggs):
            raise ValueError(f'agg must be one of {valid_aggs}. Got: {agg}')


class MetricSpecDict(TypedDict):
    key_path: str
    class_idxs: NotRequired[Optional[IndexLike]]
    agg: NotRequired[Optional[Aggregation]]
    unit: NotRequired[Optional[str]]

MetricSpecLike: TypeAlias = Union[MetricSpec, MetricSpecDict]


#####################################
# Functions
#####################################
def normalize_measure(value: MeasureValue) -> MeasureValue:
    '''
    Normalizes a measure value (e.g. loss or metric) into a format suitable for tasks like logging and saving.
    
    The procedure is as follows:
        - Detach tensors from computational graph (they should ideally be already detached).
        - Moves multi-element tensors to CPU.
        - Converts single-element tensors to Python floats.

    Args:
        value (MeasureValue): Computed measure value (a float or tensor).

    Returns:
        MeasureValue: Normalized measure value (a tensor or float).
    '''
    if isinstance(value, torch.Tensor):
        value = value.detach()
        return float(value.item()) if value.numel() == 1 else value.cpu()
    
    elif isinstance(value, float):
        return value
    
    else:
        raise TypeError(f'Expected type float of torch.Tensor. Got {type(value)}.')
    

def normalize_metric_spec(spec: MetricSpecLike) -> MetricSpec:
    '''
    Normalizes a metric specifications to a `MetricSpec` instance.
    If input is already a `MetricSpec` instance, it is returned unchanged.

    Args:
        spec (List[MetricSpecLike]): A metric specifications. 
                                     This is either a `MetricSpec` instance
                                     or a `MetricSpecDict` dictionary.
    Returns:
        MetricSpec: `MetricSpec` instance constructed from `spec`.
    '''
    if isinstance(spec, dict):
        return MetricSpec(**spec)
    elif isinstance(spec, MetricSpec):
        return spec
    else:
        raise TypeError(
            'spec must be a MetricSpec instance or a MetricSpecDict dictionary.'
        )
    

def select_and_agg_metric(
    metric_data: Union[MetricResults, MetricSeriesResults],
    key_path: str,
    class_idxs: Optional[IndexLike] = None,
    agg: Optional[Aggregation] = None
) -> Union[MeasureValue, MeasureSeries]:
    '''
    Extract a metric from a possibly nested result dictionary using a dot-separated key path.

    If the metric is a tensor:
        - Optionally subset it based on class indices
        - Optionally aggregate it across all dimensions (e.g. mean, median, min, max)

    If the metric is a list of tensors, this procedure is applied independently to each tensor.

    Args:
        metric_data (Union[MetricResults, MetricSeriesResults]): Result dictionary to extract from.
                                                                 If nested, the maximum depth is 2.
                                                                 All leaf values of this dictionary must be
                                                                 a float, tensor, or a list of floats/tensors.
        key_path (str): Dot-separated key path consisting of only keys from `metric_data`.
        class_idxs (optional, IndexLike): Class indices used to subset tensors in the extracted metric.
                                          If not provided, no subsetting is applied.
        agg (optional, Aggregation): Aggregation method applied to tensors in the 
                                     extracted metric after optional class subsetting.
                                     If provided, must be one of: 'mean', 'median', 'min', 'max'.
                                     If not provided, no aggregation is applied.

    Returns:
        Union[MeasureValue, MeasureSeries]: Metric extracted from `key_path`, 
                                            with optional class subsetting and aggregation.
                                            The data type matches the raw extracted metric in `metric_data`.
                                            If both `class_idxs` and `agg` are `None`, 
                                            a shallow copy of the raw extracted metric is returned.

    '''
    metric_values = nested_extract(metric_data, key_path) # Extract metric
    is_series = isinstance(metric_values, list)

    if (class_idxs is None) and (agg is None):
        return metric_values.copy() if is_series else metric_values
    
    # Optional subsetting and aggregation (For tensors)
    metric_values = metric_values if is_series else [metric_values]
    processed_values = []
    for value in metric_values:
        if isinstance(value, torch.Tensor):
            if class_idxs is not None:
                value = value[class_idxs] # Subset based on class indices

            if agg is not None:
                value = apply_agg(value, agg) # Aggregate; stays a tensor

        else:
            raise TypeError(
                'If class_idxs and/or agg is provided, '
                f"the extracted metric at key_path '{key_path}' "
                f'must only contain tensors. Got: {type(value)}'
            )
        processed_values.append(value)

    return processed_values if is_series else processed_values[0]


def select_and_agg_scalar_metric(
    metric_data: Union[MetricResults, MetricSeriesResults],
    key_path: str,
    class_idxs: Optional[IndexLike] = None,
    agg: Optional[Aggregation] = None
) -> Union[float, List[float]]:
    '''
    Equivalent to `select_and_agg_metric`, but enforces that all values 
    in the result are converted to Python floats.

    If `select_and_agg_metric` returns a float or list of floats, the result is returned unchanged.
    If `select_and_agg_metric` returns a tensor or list of tensors, 
    each tensor must be single-element and is converted to a float using `.item()`.

    Args:
        metric_data (Union[MetricResults, MetricSeriesResults]): Result dictionary to extract from.
                                                                 If nested, the maximum depth is 2.
                                                                 All leaf values of this dictionary must be
                                                                 a float, tensor, or a list of floats/tensors.
        key_path (str): Dot-separated key path consisting of only keys from `metric_data`.
        class_idxs (optional, IndexLike): Class indices used to subset tensors in the extracted metric.
                                          If not provided, no subsetting is applied.
        agg (optional, Aggregation): Aggregation method applied to tensors in the 
                                     extracted metric after optional class subsetting.
                                     If provided, must be one of: 'mean', 'median', 'min', 'max'.
                                     If not provided, no aggregation is applied.

    Returns:
        Union[float, List[float]]: Extracted metric from `key_path`, with optional class subsetting and aggregation.
                                   Each tensor in the result is converted to a Python float.
    '''
    metric_values = select_and_agg_metric(metric_data, key_path, class_idxs, agg)

    is_series = isinstance(metric_values, list)
    metric_values = metric_values if is_series else [metric_values]

    for i, value in enumerate(metric_values):
        if isinstance(value, float):
            continue

        elif value.numel() == 1:
            # value must be a tensor here
            metric_values[i] = value.item()

        else:
            raise ValueError(
                'Expected the metric produced by the arguments '
                f"(key_path='{key_path}', class_idxs={class_idxs}, agg={agg}) "
                'to only contain floats or single-element tensors. '
                f'Got a tensor with {value.numel()} elements.'
            )
        
    return metric_values if is_series else metric_values[0]


