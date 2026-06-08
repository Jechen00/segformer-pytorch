#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torch.amp import GradScaler
from torch.optim import Optimizer, lr_scheduler

from pathlib import Path
from typing import Union, Optional, Dict, Any, Literal, Iterable, TypeAlias, get_args

from src.logging.history import TrainHistory, ValHistory
from src.engine.measure_policy import MeasurePolicy
from src.utils import file_utils, data_utils, ml_utils

Components: TypeAlias = Literal['model', 'optimizer', 'scaler', 'scheduler', 'measure_policy', 'histories']
ComponentInput: TypeAlias = Union[Components, Iterable[Components]]
ALL_COMPONENTS = set(get_args(Components))


#####################################
# Functions
#####################################
def save_checkpoint(
    model: nn.Module, 
    optimizer: Optimizer, 
    scaler: GradScaler,
    train_history: TrainHistory,
    val_history: ValHistory,
    checkpoint_epoch: int,
    save_path: Union[str, Path],
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    measure_policy: Optional[MeasurePolicy] = None
) -> None:
    '''
    Saves a checkpoint containing the state dicts for:
        - Model
        - Optimizer
        - Gradient scaler
        - Training history (loss average)
        - Validation history (loss average and metrics)
        - (Optional) learning rate scheduler
        - (Optional) measure policy

    Additionally saves the epoch index of the checkpoint.

    Args:
        model (nn.Module): 
            Model to save.
        optimizer (Optimizer): 
            Optimizer for the `model`.
        scaler (GradScaler): 
            Gradient scaler for automatic mixed precision (AMP).
            Required because the training system always has a scaler instance,
            even if AMP is disabled (e.g. CPU/MPS).
        train_history (TrainHistory): 
            Training dataset history containing loss avevalues across epochs.
        val_histories (ValHistory): 
            Validation dataset history containing loss and metric values across epochs.
        checkpoint_epoch (int): 
            Index of the completed epoch this checkpoint refers to.
            Note that this may differ from the scheduler's internal step counter called `last_epoch`.
        save_path (Union[str, Path]): 
            Full path to save the checkpoint to. 
            This should end with a file extension (e.g. '.pt' or '.pth').
        scheduler (optional, lr_scheduler._LRScheduler): 
            Learning rate scheduler for the `optimizer`.
        measure_policy (optional, MeasurePolicy): 
            Measure policy for early stopping and best score tracking.
    '''

    # Validate save_path and create directory if it doesn't exist
    save_path = file_utils.format_file_path(save_path, 'save_path')
    save_path.parent.mkdir(parents = True, exist_ok = True)

    # Create checkpoint dictionary and save
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'scheduler': None if scheduler is None else scheduler.state_dict(),
        'measure_policy': None if measure_policy is None else measure_policy.state_dict(),
        'train_history': train_history.state_dict(),
        'val_history': val_history.state_dict(),
        'checkpoint_epoch': checkpoint_epoch
    }

    torch.save(obj = checkpoint, f = save_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    train_history: Optional[TrainHistory] = None,
    val_history: Optional[ValHistory] = None,
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    measure_policy: Optional[MeasurePolicy] = None,
    device: Union[str, torch.device] = 'cpu'
) -> int:
    '''
    Loads a saved training checkpoint from `checkpoint_path`.

    Note: The checkpoint must have a `model` key to load the model state dict.
          If an optional argument is provided, the checkpoint must contain the corresponding state dict
          under a key of the same name (e.g. `optimizer` state dict must be stored under the `optimizer` key).

    Args:
        checkpoint_path (Union[str, Path]): 
            Full path to a checkpoint file to load a checkpoint.
        model (nn.Module): 
            Model to load the state_dict from `checkpoint['model']`.
            This should already be on `device`.
        optimizer (optional, Optimizer): 
            Optimizer for `model`.
        scaler (optional, GradScaler): 
            Gradient scaler for automatic mixed precision (AMP).
        train_history (optional, TrainHistory): 
            Training dataset history containing loss values across epochs.
            If provided, the checkpoint state_dict is always moved to CPU before loading.
        val_history (optional, ValHistory): 
            Validation dataset history containing loss and metric values across epochs.
            If provided, the checkpoint state_dict is always moved to CPU before loading.
        scheduler (optional, lr_scheduler._LRScheduler): 
            Learning rate scheduler for `optimizer`.
        measure_policy (optional, MeasurePolicy): 
            Measure policy for early stopping and best score tracking.
        device (Union[str, torch.device]): 
            The device to load the checkpoint tensors on to. Default is 'cpu'.

    Returns:
        int:  
            Index of the completed epoch the checkpoint was saved at.
            This may differ from the internal step counter in `scheduler`, called `last_epoch`.
    '''
    checkpoint = torch.load(checkpoint_path, map_location = device)

    training_comps = {
        'model': model,
        'optimizer': optimizer,
        'scaler': scaler,
        'scheduler': scheduler,
        'measure_policy': measure_policy
    }

    history_comps = {
        'train_history': train_history,
        'val_history': val_history
    }

    valid_training_keys = [key for (key, obj) in training_comps.items() 
                           if obj is not None]
    valid_history_keys = [key for (key, obj) in history_comps.items()
                          if obj is not None]
    
    # Check for any missing keys in checkpoint
    keys_to_check = set(valid_training_keys) | set(valid_history_keys) | {'checkpoint_epoch'}
    missing = keys_to_check  - checkpoint.keys()
    if missing:
        raise KeyError(f'Missing keys in checkpoint: {missing}')
    
    # Load state dicts
    for key in valid_training_keys:
        training_comps[key].load_state_dict(checkpoint[key])
    
    for key in valid_history_keys:
        history_comps[key].load_state_dict(ml_utils.recursive_to_cpu(checkpoint[key]))

    return checkpoint['checkpoint_epoch']


def separate_checkpoint(
    checkpoint: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    components: Optional[ComponentInput] = None,
    base_dir: Optional[Union[str, Path]] = None
) -> None:
    '''
    Extracts components from a checkpoint and saves them as separate `.pth` files.

    Notes: 
        - Exactly one of `checkpoint` or `checkpoint_path` must be provided.
        - With the exception of `histories`, the checkpoint must have a matching key for each element in `components`.
        - If `histories` is in `components`, the checkpoint must have the keys `train_history` and `val_history`.
        - The checkpoint must always have the key `checkpoint_epoch`.

    Args:
        checkpoint (optional, Dict[str, Any]): 
            Checkpoint dictionary to separate.
        checkpoint_path (optional, Union[str, Path]): 
            Path to a checkpoint file to load and then separate.              
        components (optional, ComponentInput): 
            The component(s) to extract and save from the checkpoint.  
            This can be a single string or an iterable of strings.
            If not provided, defaults to all valid components:
                {'model', 'optimizer', 'scaler', 
                    'scheduler', 'measure_policy', 'histories'}
        base_dir (optional, Union[str, Path]): 
            The base directory where the save directory will be created.
            The save directory is `base_dir/checkpoint_epoch_{num}`,
            where `num = checkpoint['checkpoint_epoch']`.
    '''
    # Check provided checkpoint and load it in if needed
    if data_utils.all_or_none(checkpoint, checkpoint_path):
        raise ValueError('Exactly one of checkpoint and checkpoint_path must be provided.')
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')

    # Format components into a set
    if components is None:
        components = ALL_COMPONENTS
    elif isinstance(components, str):
        components = {components}
    else:
        components = set(components)

    sep_histories = 'histories' in components # Whether to separate histories
    training_components = components - {'histories'}

    # Check for invalid components
    invalid = components - ALL_COMPONENTS
    if invalid:
        raise ValueError(f'Unsupported components: {invalid}')

    # Check for missing keys in checkpoint
    keys_to_check = training_components | {'checkpoint_epoch'}
    if sep_histories:
        keys_to_check.update(['train_history', 'val_history'])

    missing = keys_to_check  - checkpoint.keys()
    if missing:
        raise KeyError(f'Missing keys in checkpoint: {missing}')

    # Make directory to save separate files
    base_dir = Path(base_dir) if base_dir is not None else Path()
    save_dir = base_dir / f"checkpoint_epoch_{checkpoint['checkpoint_epoch']}"
    save_dir.mkdir(parents = True, exist_ok = True)

    # Save components
    for key in training_components:
        torch.save(checkpoint[key], save_dir / f'{key}.pth')

    if sep_histories:
        histories = {
            'train_history': checkpoint['train_history'],
            'val_history': checkpoint['val_history']
        }
        torch.save(histories, save_dir / 'histories.pth')