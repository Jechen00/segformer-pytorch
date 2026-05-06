#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from pathlib import Path
from typing import Union, Optional, Dict, Any, Literal, Iterable, TypeAlias

from src.logging.history import TrainHistory, ValHistory
from src.engine.measure_policy import MeasurePolicy
from src.utils import recursive_to_cpu, normalize_file_path, all_or_none

Components: TypeAlias = Literal['model', 'optimizer', 'scheduler', 'measure_policy', 'histories']
ComponentInput: TypeAlias = Union[Components, Iterable[Components]]
ALL_COMPONENTS = {'model', 'optimizer', 'scheduler', 'measure_policy', 'histories'}


#####################################
# Functions
#####################################
def save_checkpoint(
    model: nn.Module, 
    optimizer: Optimizer, 
    train_history: TrainHistory,
    val_history: ValHistory,
    checkpoint_epoch: int,
    save_path: Union[str, Path],
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    measure_policy: Optional[MeasurePolicy] = None
) -> None:
    '''
    Saves a checkpoint containing the model, optimizer, (optional) scheduler state dicts.
    Also saves training history (loss average) and validation history (loss average and metrics) 
    along with the epoch index.

    Args:
        model (nn.Module): Model to save.
        optimizer (Optimizer): Optimizer for the `model`.
        train_history (TrainHistory): Training dataset history containing loss avevalues across epochs.
        val_histories (ValHistory): Validation dataset history containing loss and metric values across epochs.
        checkpoint_epoch (int): Index of the completed epoch this checkpoint refers to.
                                Note that this may differ from the scheduler's internal step counter called `last_epoch`.
        save_path (Union[str, Path]): Full path to save the checkpoint to. 
                                      This should end with a file extension (e.g. '.pt' or '.pth').
        scheduler (optional, lr_scheduler._LRScheduler): Learning rate scheduler for the `optimizer`.
    '''

    # Validate save_path and create directory if it doesn't exist
    save_path = normalize_file_path(save_path, 'save_path')
    save_path.parent.mkdir(parents = True, exist_ok = True)

    # Create checkpoint dictionary and save
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
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
    optimizer: Optimizer,
    train_history: TrainHistory,
    val_history: ValHistory,
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    measure_policy: Optional[MeasurePolicy] = None,
    device: Union[str, torch.device] = 'cpu'
) -> int:
    '''
    Loads a saved training checkpoint from `checkpoint_path`.

    The checkpoint is expected to match the format produced by `checkpoint.save_checkpoint()`.
    Checkpoint keys expected:
        - 'model'
        - 'optimizer'
        - 'train_history'
        - 'val_history'
        - 'scheduler'
        - 'measure_policy'
        - 'checkpoint_epoch'

    Args:
        checkpoint_path (Union[str, Path]): Full path to a checkpoint file to load a checkpoint.
        model (nn.Module): Model to load the state_dict from `checkpoint['model']`.
                           This should already be on device.
        optimizer (Optimizer): Optimizer for model to load the state_dict from `checkpoint[optimizer]`.
        train_history (TrainHistory): Training dataset history containing loss values across epochs.
                                      The state_dict in `checkpoint['train_history']` is 
                                      always moved to CPU before loading into this PhaseHistory.
        val_history (ValHistory): Validation dataset history containing loss and metric values across epochs.
                                    The state_dict in `checkpoint['val_history']` is 
                                    always moved to CPU before loading into this PhaseHistory.
        scheduler (optional, lr_scheduler._LRScheduler): Learning rate scheduler for the optimizer.
                                                         If provided, an existing (non-None) scheduler state_dict
                                                         must be stored in the `checkpoint['scheduler']`.
        device (Union[str, torch.device]): The device to load the checkpoint tensors on to. Default is 'cpu'.

    Returns:
        int:  Index of the completed epoch the checkpoint was saved at.
              Note that this may differ from the scheduler's internal step counter called `last_epoch`.
    '''
    checkpoint = torch.load(checkpoint_path, map_location = device)

    # Load model and optimizer
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Load training and validation history (with tensors on CPU)
    train_history.load_state_dict(recursive_to_cpu(checkpoint['train_history']))
    val_history.load_state_dict(recursive_to_cpu(checkpoint['val_history']))

    # Load scheduler
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    # Load measure policy
    if measure_policy is not None:
        measure_policy.load_state_dict(checkpoint['measure_policy'])

    return checkpoint['checkpoint_epoch']

def separate_checkpoint(
    checkpoint: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    components: Optional[ComponentInput] = None,
    base_dir: Optional[Union[str, Path]] = None
) -> None:
    '''
    Extracts components from a checkpoint and saves them as separate `.pth` files.

    Args:
        checkpoint (optional, Dict[str, Any]): Checkpoint dictionary to separate.
        checkpoint_path (optional, Union[str, Path]): Path to a checkpoint file to load and then separate.              
        components (optional, ComponentInput): The component(s) to extract and save from the checkpoint.  
                                               This can be a single string or an iterable of strings.
                                               If not provided, defaults to all valid components:
                                                    {'model', 'optimizer', 'scheduler', 'measure_policy', 'histories'}
        base_dir (optional, Union[str, Path]): The base directory where the save directory will be created.
                                               The save directory is `base_dir/checkpoint_epoch_{num}`,
                                               where `num = checkpoint['checkpoint_epoch']`.
    Notes: 
        - Exactly one of `checkpoint` or `checkpoint_path` must be provided.
        - With the exception of `histories`, the checkpoint must have a matching key for each element in `components`.
        - If `histories` is in `components`, the checkpoint must have the keys `train_history` and `val_history`.
        - The checkpoint must always have the key `checkpoint_epoch`.
    '''
    # Check provided checkpoint and load it in if needed
    if all_or_none(checkpoint, checkpoint_path):
        raise ValueError('Exactly one of checkpoint and checkpoint_path must be provided.')
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location = 'cpu')

    # Normalize components into a set
    if components is None:
        components = ALL_COMPONENTS
    elif isinstance(components, str):
        components = {components}
    else:
        components = set(components)

    # Check for invalid components
    invalid = components - ALL_COMPONENTS
    if len(invalid) != 0:
        raise ValueError(f'Unsupported components: {invalid}')

    # Check for missing components
    keys_to_check = (components - {'histories'}) | {'checkpoint_epoch'}
    if 'histories' in components:
        keys_to_check.update(['train_history', 'val_history'])

    missing = [key for key in keys_to_check if key not in checkpoint]
    if len(missing) != 0:
        raise KeyError(f'Missing keys in checkpoint: {missing}')

    # Make directory to save separate files
    base_dir = Path(base_dir) if base_dir is not None else Path()
    save_dir = base_dir / f"checkpoint_epoch_{checkpoint['checkpoint_epoch']}"
    save_dir.mkdir(parents = True, exist_ok = True)

    # Save components
    for key in components:
        if key == 'histories':
            histories = {
                'train_history': checkpoint['train_history'],
                'val_history': checkpoint['val_history']
            }
            torch.save(histories, save_dir / 'histories.pth')
        else:
            torch.save(checkpoint[key], save_dir / f'{key}.pth')