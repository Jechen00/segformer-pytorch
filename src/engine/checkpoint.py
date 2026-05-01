#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler

from pathlib import Path
from typing import Union, Optional

from src.logging.history import PhaseHistory
from src.engine.measure_policy import MeasurePolicy
from src.utils import recursive_to_cpu, normalize_file_path


#####################################
# Functions
#####################################
def save_checkpoint(
    model: nn.Module, 
    optimizer: Optimizer, 
    train_history: PhaseHistory,
    val_history: PhaseHistory,
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
        train_history (PhaseHistory): Training dataset history containing loss avevalues across epochs.
        val_histories (PhaseHistory): Validation dataset history containing loss and metric values across epochs.
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
    train_history: PhaseHistory,
    val_history: PhaseHistory,
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
        train_history (PhaseHistory): Training dataset history containing loss values across epochs.
                                      The state_dict in `checkpoint['train_history']` is 
                                      always moved to CPU before loading into this PhaseHistory.
        val_history (PhaseHistory): Validation dataset history containing loss and metric values across epochs.
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