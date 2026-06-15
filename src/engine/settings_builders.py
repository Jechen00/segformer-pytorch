#####################################
# Imports & Dependencies
#####################################
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from typing import Dict, Any, Literal, Optional, Union, List, TypeAlias

from src.engine.trainer_settings import SchedulerSettings, PerformanceSettings
from src.ml_types import PythonNum

NumOrList: TypeAlias = Union[PythonNum, List[PythonNum]]


#####################################
# Mappings
#####################################
MEMORY_FORMATS = {
    'channels_last': torch.channels_last,
    'contiguous_format': torch.contiguous_format
}

AMP_DTYPES = {
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}


#####################################
# Functions
#####################################
def make_sched_settings(
    optimizer: Optimizer,
    sched_class: type[LRScheduler], 
    step_freq: Literal['epoch', 'optim_step'],
    sched_static_args: Optional[Dict[str, Any]] = None, 
    sched_timing_args: Optional[Dict[str, NumOrList]] = None, 
    num_optim_steps: Optional[int] = None
) -> SchedulerSettings:
    '''
    Instantiates a `SchedulerSettings` dataclass.
    This includes creating a learning rate scheduler from
    the provided class and optional argument dictionaries.
    If the scheduler step frequency is per optimizer step (`step_freq='optim_step'`),
    any provided timing-related arguments will be multiplied by `num_optim_steps`.

    Notes on Scheduler Arguments:
        - All required arguments for the scheduler class (`sched_class`)
          must be provided through eitehr `sched_static_args` or `sched_timing_args`.

    Args:
        optimizer (Optimizer):
            Optimizer associated with the scheduler.
        sched_class (type[LRScheduler]):
            Learning rate scheduler class to instantiate (should not already be an instance).
        step_freq (Literal['epoch', 'optim_step']):
            Step/update frequency for the scheduler.
                - epoch: The scheduler is stepped/updated once per epoch,
                         after the training loop finishes.
                - optim_step: The scheduler is stepped/updated after each optimizer step.
        sched_static_args (optional, Dict[str, Any]):
            Dictionary of scheduler arguments that are not timing related.
            If provided, these are arguments are always passed to `sched_class` unchanged.
            They are not multiplied by `num_optim_steps` if `step_freq='optim_step'`.
        sched_timing_args (optional, Dict[str, NumOrList]):
            Dictionary of scheduler arguments that are timing related.
            The units of these arguments should be epoch-based.
            If provided and `step_freq='optim_step'`,
            these arguments are multiplied by `num_optim_steps` 
            (converting to optimizer-based units) before being passed to `sched_class`.
        num_optim_steps (optional, int):
            Number of optimizer steps per epoch.
            This must be provided if `sched_timing_args` is provided and `step_freq='optim_step'`.
    Returns:
        SchedulerSettings:
            The `SchedulerSettings` instance containing 
            the constructed scheduler and its step frequency.
    '''
    sched_static_args = sched_static_args.copy() if sched_static_args is not None else {}
    sched_timing_args = sched_timing_args.copy() if sched_timing_args is not None else {}

    if (step_freq == 'optim_step') and sched_timing_args:
        if num_optim_steps is None:
            raise ValueError(
                "If step_freq = 'optim_step', num_optim_steps must be provided."
            )
        
        for key, value in sched_timing_args.items():
            if isinstance(value, PythonNum):
                sched_timing_args[key] = value * num_optim_steps
            elif isinstance(value, list):
                sched_timing_args[key] = [t * num_optim_steps for t in value]
            else:
                raise TypeError(
                    'All values in sched_timing_args must be an integer/float or a list of integers/floats.'
                )

    sched_cfg = sched_static_args | sched_timing_args

    return SchedulerSettings(
        scheduler = sched_class(optimizer, **sched_cfg),
        step_freq = step_freq
    )


def make_perf_settings(config: Dict[str, Any]) -> PerformanceSettings:    
    '''
    Instantiates a `PerformanceSettings` dataclass from a config dictionary.

    Args:
        config (Dict[str, Any]):
            Config dictionary containing only arguments for `PerformanceSettings`.
            If `memory_format` and `amp_dtype` are present,
            their values must be keys in the `MEMORY_FORMATS` and `AMP_DTYPES`
            mappings, respectively.

    Returns:
        PerformanceSettings:
            The `PerformanceSettings` instance constructed from `config`.
    '''
    config = config.copy()
    
    if 'memory_format' in config:
        config['memory_format'] = MEMORY_FORMATS[config['memory_format']]
    
    if 'amp_dtype' in config:
        config['amp_dtype'] = AMP_DTYPES[config['amp_dtype']]

    return PerformanceSettings(**config)