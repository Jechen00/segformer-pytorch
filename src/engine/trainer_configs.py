#####################################
# Imports & Dependencies
#####################################
from torch.optim import lr_scheduler

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Literal, Union

from src.metrics.ops import Metric
from src.metrics.postprocess import MetricSpecLike, normalize_metric_spec
from src.utils import normalize_file_path


#####################################
# Data Classes
#####################################
@dataclass
class SchedulerConfig():
    scheduler: lr_scheduler._LRScheduler
    step_freq: Literal['epoch', 'optim_step'] = 'optim_step'


@dataclass
class EvalConfig():
    '''
    For validation only.
    '''
    metrics: Dict[str, Metric]
    eval_interval: int
    log_metric_specs: Optional[Dict[str, MetricSpecLike]] = None
    
    def __post_init__(self):   
        # Check value for eval_interval
        if self.eval_interval < 1:
            raise ValueError(f'eval_interval must be at least 1 if provided.')
        
        # Normalize metric specifications (for logging) if provided
        log_metric_specs = self.log_metric_specs
        if log_metric_specs is not None:
            for name, spec in log_metric_specs.items():
                log_metric_specs[name] = normalize_metric_spec(spec)


@dataclass
class SaveConfig():
    save_dir: Optional[Union[str, Path]] = None
    ckpt_name: Optional[Union[str, Path]] = None
    best_model_name: Optional[Union[str, Path]] = None
    ignore_exists: bool = False

    def __post_init__(self):
        for name_attr in ['ckpt_name', 'best_model_name']:
            name = getattr(self, name_attr)
            if name is None:
                continue
            elif self.save_dir is None:
                raise ValueError(f'save_dir must be provided if {name_attr} is provided.')

            # Check that name is a proper file name
            p_name = normalize_file_path(name, name_attr)
            if len(p_name.parts) != 1:
                raise ValueError(f'{name_attr} must be a single file name. Got: {name}')

            # Check that the path will not unintentionally overwrite a file
            name_path = Path(self.save_dir) / name
            if (not self.ignore_exists) and (name_path.exists()):
                raise FileExistsError(
                    f'A file already exists at save_dir/{name_attr}: {str(name_path)}. '
                    'To allow overwriting this file, set ignore_exists = True.'
                )
            
    @property
    def ckpt_path(self) -> Optional[Path]:
        if self.ckpt_name is None:
            return None
        return Path(self.save_dir) / self.ckpt_name

    @property
    def best_model_path(self) -> Optional[Path]:
        if self.best_model_name is None:
            return None
        return Path(self.save_dir) / self.best_model_name
    

@dataclass
class LogConfig():
    logbox_len: int = 100
    num_decimals: int = 4
    max_row_entries: int = 3