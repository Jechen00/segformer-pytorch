#####################################
# Imports & Dependencies
#####################################
import torch

from typing import List, Union, TypeAlias, Dict

#####################################
# Types
#####################################
MeasureValue: TypeAlias = Union[float, torch.Tensor]
MetricGroup: TypeAlias = Dict[str, MeasureValue]
'''
Dictionary of grouped metric values (each value is a `MeasureValue`).
Example layout:
    {
        'metric_1': value,
        'metric_2': value
    }
where `metric_x` is the metric name and `value` is a float or tensor.
'''

MetricResults: TypeAlias = Dict[str, Union[MeasureValue, MetricGroup]]
'''
Dictionary of metric results, where each value is a `MeasureValue` or a `MetricGroup`.
This may be nested and have a maximum depth of 2.
Example layout:
    {
        'task_1': {
            'metric_1': value,
            'metric_2': value
        },
        'task_2': {
            'metric_1': value,
            'metric_2': value
        }
        'task_3': value
    }
where `task_x` is the evaluation task (e.g. accuracy, segmentation metrics, classification metrics),
`metric_x` is the metric name, and `value` is a float or tensor.
'''

MeasureResult: TypeAlias = Union[MeasureValue, MetricGroup, MetricResults]

MeasureSeries: TypeAlias = Union[List[float], List[torch.Tensor]]
MetricSeriesGroup: TypeAlias = Dict[str, MeasureSeries]
'''
Dictionary of grouped metric value lists (each value is a `MeasureSeries`).
Each list should ideally be the same length.
Example layout:
    {
        'metric_1': [value_1, value_2, ..., value_k],
        'metric_2': [value_1, value_2, ..., value_k]
    }
where `metric_x` is the metric name and `value_x` is a float or tensor.
'''

MetricSeriesResults: TypeAlias = Dict[str, Union[MeasureSeries, MetricSeriesGroup]]
'''
Dictionary of metric result lists, where each value is a `MeasureSeries` or a `MetricSeriesGroup`.
Each list should ideally be the same length.
This dictionary may be nested and have a maximum depth of 2.
Example layout:
    {
        'task_1': {
            'metric_1': [value_1, value_2, ..., value_k],
            'metric_2': [value_1, value_2, ..., value_k]
        },
        'task_2': {
            'metric_1': [value_1, value_2, ..., value_k],
            'metric_2': [value_1, value_2, ..., value_k]
        }
        'task_3': [value_1, value_2, ..., value_k]
    }
where `task_x` is the evaluation task (e.g. accuracy, segmentation metrics, classification metrics),
`metric_x` is the metric name, and `value` is a float or tensor.
'''