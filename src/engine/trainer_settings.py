'''
Setting dataclasses used in `src.trainer.ModelTrainer`.

Each dataclass in this module corresponds to an argument 
in `src.trainer.ModelTrainer` and is used 
to configure a certain part of the training process.

This includes:
    - Learning rate scheduling
    - Metric evaluation on the validation dataset
    - Training performance settings (device, memory format, and AMP)
    - Checkpoint and best-model save paths
    - Log formatting style
'''


#####################################
# Imports & Dependencies
#####################################
import torch
from torch.optim.lr_scheduler import LRScheduler

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Literal, Union

from src.metrics.ops import Metric
from src.metrics.postprocess import MetricSpecLike, format_metric_spec

from src.utils.file_utils import format_file_path


#####################################
# Data Classes
#####################################
@dataclass
class SchedulerSettings():
    '''
    Settings for the learning rate scheduler.

    Attributes:
        scheduler (LRScheduler):
            The learning rate scheduler to use during training.
        step_freq (Literal['epoch', 'optim_step']):
            Step/update frequency for `scheduler`.
                - epoch: The scheduler is stepped/updated once per epoch,
                         after the training loop finishes.
                - optim_step: The scheduler is stepped/updated after each optimizer step.
    '''
    scheduler: LRScheduler
    step_freq: Literal['epoch', 'optim_step']


@dataclass
class EvalSettings():
    '''
    Settings for validation metric evaluation.

    Attributes:
        metrics (Dict[str, Metric]):
            Dictionary mapping task names to metric objects used during evalutation.
            Each metric object must implement the `Metric` protocol (see `src.metrics.ops` for details).
            Example:
                {
                    'cls': `src.metrics.ops.ClassificationMetrics(...)`,
                    'seg': `src.metrics.ops.SegmentationMetrics(...)`,
                    'acc': `torchmetrics.Accuracy()`
                }
        eval_interval (int):
            Interval (in epochs) to compute evaluation metrics on the validation dataset.
            The first evalutation interval always starts **after** epoch 0.
        log_metric_specs (optional, Dict[str, MetricSpecLike]):
            Dictionary mapping metric names to metric specifications.
            Each specification defines how a metric is extracted and 
            optionally subsetted and/or aggregated for logging.

            Supported specification types:
                - `MetricSpec` instance.
                - `MetricSpecDict` dictionary.

            Ensure that the `key_path` in each specification contains a valid key path from `metrics`.
            Example:
                {
                    'Accuracy': MetricSpec(key_path = 'acc'),
                    'Dice (Mean)':  MetricSpec(key_path = 'seg.mean_dice'),
                    'IoU (Mean)': MetricSpec(key_path = 'seg.mean_iou'),
                    'Dice (Min)': MetricSpec(key_path = 'seg.dice', agg = 'min'),
                    'IoU (Min)': MetricSpec(key_path = 'seg.iou', agg = 'min'),
                    'Dice (Human)': MetricSpec(key_path = 'seg.dice', class_idxs = 1),
                    'IoU (Human)': MetricSpec(key_path = 'seg.iou', class_idxs = 1)
                }
    '''
    metrics: Dict[str, Metric]
    eval_interval: int
    log_metric_specs: Optional[Dict[str, MetricSpecLike]] = None
    
    def __post_init__(self):   
        # Check value for eval_interval
        if self.eval_interval < 1:
            raise ValueError(f'eval_interval must be at least 1 if provided.')
        
        # Format metric specifications (for logging) if provided
        log_metric_specs = self.log_metric_specs
        if log_metric_specs is not None:
            for name, spec in log_metric_specs.items():
                log_metric_specs[name] = format_metric_spec(spec)


@dataclass
class PerformanceSettings():
    '''
    Settings for training performance

    Attributes:
        device (Union[str, torch.device]):
            The device to perform computations on. 
            Default is 'cpu'.
        memory_format (torch.memory_format):
            The memory format to use for the model and input tensors during training.
            Depending on the device (typically CUDA), using `torch.channels_last` 
            may improve training performance for convolution-heavy models (like SegFormer).
            Default is `torch.contiguous_format`.
        use_amp (bool):
            Whether to use Automatic Mixed Precision (AMP).
            This requires the device type to be CUDA.
            For details on AMP, see: https://docs.pytorch.org/docs/2.12/amp.html
            Default is `False`.
        amp_dtype (torch.dtype):
            The floating-point datatype to use for AMP autocasting.
            Default is `torch.float16`.
    '''
    device: Union[str, torch.device] = 'cpu'
    memory_format: torch.memory_format = torch.contiguous_format # torch.channels_last improves training time on GPU
    use_amp: bool = False
    amp_dtype: torch.dtype = torch.float16

    def __post_init__(self):
        device = torch.device(self.device)
        device_type = device.type

        # Check device type is valid
        if device_type == 'cpu':
            pass

        elif device_type == 'mps':
            if not torch.backends.mps.is_available():
                raise ValueError('MPS is not available.')
            
        elif device_type == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA is not available.')

        else:
            raise ValueError(f'Unsupported device type: {device_type}')

        # Check that device is CUDA if AMP is enabled
        if self.use_amp and device_type != 'cuda':
            raise ValueError(
                f'AMP is only supported for CUDA devices. Got: {device}'
            )
        
        self.device = device

@dataclass
class SaveSettings():
    '''
    Settings for save paths of the training checkpoint 
    and the best-model state dictionary.

    Notes on Filename Arguments:
        - Instantiating this dataclass means that saving is intended during training.
          Therefore, at least one of `ckpt_name` (for checkpoint saving) 
          or `best_model_name` (for best-model saving) must be provided.

        - `ckpt_name` and `best_model_name` must be filenames 
           that end with an extension (e.g. `checkpoint.pth`).
           They must not include directory paths (e.g. `dir_1/dir_2/checkpoint.pth`).

    Notes on Training Checkpoint:
        - If checkpoint saving is enabled, the checkpoint file
          is always updated at the end of each epoch 
          to store the latest training state.

        - See `src.checkpoint.save_checkpoint` for details 
          on what the training checkpoint contains.

    Notes on Best Model Saving:
        - Saving a best-model state dictionary requires 
          a measure policy (`src.meaure_policy.MeasurePolicy`)
          to be provided in the `measure_policy` argument of 
          the same `src.trainer.ModelTrainer` instance.
          This defines the score used to determine the best model.

        - The best model refers to the model that achieves
          the current best score in the measure policy.
          Its state dictionary is saved whenever the best score improves.

    Attributes:
        save_dir (Union[str, Path]):
            Directory for saving training files (checkpoint and best model).
        ckpt_name (optional, Union[str, Path]):
            Filename used when saving the training checkpoint.
            If provided, the training checkpoint is saved to `save_dir/ckpt_name`.
            If not provided, the training checkpoint is not saved during training.
        best_model_name (optional, Union[str, Path]):
            Filename used when saving the best-model state dictionary.
            If provided, the best-model state dictionary is saved to `save_dir/best_model_name`.
            If not provided, the best-model state dictionary is not saved during training,
            even if a best score is defined with a measure policy.
        ignore_exists (bool):
            Whether to ignore existing files at `save_dir/ckpt_name` (if provided)
            and `save_dir/best_model_name` (if provided), prior to training.
            If `False` and a file already exists, a `FileExistsError` will be raised.
    '''
    save_dir: Union[str, Path]
    ckpt_name: Optional[Union[str, Path]] = None
    best_model_name: Optional[Union[str, Path]] = None
    ignore_exists: bool = False

    def __post_init__(self):
        self.save_dir = Path(self.save_dir) # Normalize to a path object

        if (self.ckpt_name is None) and (self.best_model_name is None):
            raise ValueError(
                'SaveSettings requires at least one of ckpt_name or best_model_name to be provided.'
            )

        for name_attr in ['ckpt_name', 'best_model_name']:
            name = getattr(self, name_attr)
            if name is None:
                continue

            # Check that name is a proper file name
            p_name = format_file_path(name, name_attr)
            if len(p_name.parts) != 1:
                raise ValueError(f'{name_attr} must be a single file name. Got: {name}')

            # Check that the path will not unintentionally overwrite a file
            name_path = self.save_dir / name
            if (not self.ignore_exists) and (name_path.exists()):
                raise FileExistsError(
                    f'A file already exists at save_dir/{name_attr}: {str(name_path)}. '
                    'To allow overwriting this file, set ignore_exists = True.'
                )
            
    @property
    def ckpt_path(self) -> Optional[Path]:
        '''
        Save path for the training checkpoint.
        Returns `None` if `ckpt_name` was not provided at initialization.
        '''
        if self.ckpt_name is None:
            return None
        return self.save_dir / self.ckpt_name

    @property
    def best_model_path(self) -> Optional[Path]:
        '''
        Save path for the best-model state dictionary.
        Returns `None` if `best_model_name` was not provided at initialization.
        '''
        if self.best_model_name is None:
            return None
        return self.save_dir / self.best_model_name
    

@dataclass
class LogSettings():
    '''
    Settings for training log formatting.
    See `src.logging.formatting` for details on logging.

    Attributes:
        logbox_len (int):
            Total character width of each log section.
            Default is `100`.
        num_decimals (int):
            Maximum number of decimals to display for each log value.
            Default is `4`.
        max_row_entries (int):
            Maximum number of entries in a row of each log section.
            The length of each section entry will be roughly `logbox_len//max_row_entries`.
            Default is `3`.
    '''
    logbox_len: int = 100
    max_row_entries: int = 3
    num_decimals: int = 4