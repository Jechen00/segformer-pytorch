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
MetricResults: TypeAlias = Dict[str, Union[MeasureValue, MetricGroup]]

MeasureResult: TypeAlias = Union[MeasureValue, MetricGroup, MetricResults]

MeasureSeries: TypeAlias = Union[List[float], List[torch.Tensor]]
MetricSeriesGroup: TypeAlias = Dict[str, MeasureSeries]
MetricSeriesResults: TypeAlias = Dict[str, Union[MeasureSeries, MetricSeriesGroup]]