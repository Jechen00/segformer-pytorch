#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import time
from pathlib import Path
import warnings
from typing import Optional, Union, Dict, Tuple, TypedDict

from src.engine.trainer_settings import (
    SchedulerSettings, EvalSettings, 
    SaveSettings, PerformanceSettings, LogSettings
)
from src.engine.measure_policy import MeasurePolicy
from src.engine.checkpoint import load_checkpoint, save_checkpoint

from src.logging.history import TrainHistory, ValHistory
from src.logging.formatting import (
    BOLD_ON, BOLD_OFF, EPOCH_FILL_CHAR, SEC_DIV_CHAR,
    make_epoch_header, make_metric_log_sec, make_log_sec
)

from src.metrics.ops import Metric
from src.metrics.types import MetricResults


#####################################
# Type Classes
#####################################
class ValResults(TypedDict):
    loss: torch.Tensor
    metrics: Optional[MetricResults]


#####################################
# Model Trainer Class
#####################################
class ModelTrainer():
    '''
    Implements a training pipeline for supervised learning tasks
    (e.g. image classification or semantic segmentation).

    This trains a model on a training dataset and evaluates it on a validation dataset.
    It also optionally saves a training checkpoint and best-model state dictionary.

    Task Notes:
        - The trainer predicts labels by applying `argmax` to the
          class dimension of the model logits (assumed dimension 1).
          Consequently, each target elements must only have a **single** label.
          For example:
            - Image classification must have one class label per image.
            - Semantic segmentation must have one class label per pixel.

    Dataloader Notes:
        - Each batch from the training and validation dataloaders must
          be represented as a dictionary containing:
                - image (torch.Tensor): The images for the batch.
                - targ_key (torch.Tensor): The corresponding targets for the batch.
          Here, `targ_key` is a key defined by the user when initializing `ModelTrainer`.

    Training Result Notes:
        - Epoch loss is computed as the mean of batch losses across the epoch.
          Consequently, this may not equal the mean loss across samples.

        - Computed measures are stored in:
            -  `train_history`: Epoch losses from training dataset.
            - `val_history`: Epoch losses and optional metric values from validation dataset.

    Args:
        model (nn.Module): 
            The model to train.
        train_loader (DataLoader): 
            Dataloader for the training dataset.
        val_loader (DataLoader): 
            Dataloader for the validation dataset.
        targ_key (str): 
            Dictionary key used to access the target tensor from each batch returned by the dataloaders.
        loss_fn (nn.Module):
            Loss function.
            Must accept model logits and targets (e.g. `loss_fn(logits, targs)`)
            and return a scalar value.
        optimizer (Optimizer):
            Optimizer used to update `model` weights based on the loss computed by `loss_fn`.
        sched_sett (optional, SchedulerSettings):
            Settings for the learning rate scheduler.
            If not provided, the learning rate of `optimizer` will remain constant during training.
        eval_sett (optional, EvalSettings):
            Settings for validation metric evaluations.
            If not provided, validation metrics will not be computed during training
            and `val_history` will only track the epoch loss.
        measure_policy (optional, MeasurePolicy):
            Measure policy that defines a best score (a loss or metric)
            used to determine model improvement during training.
            If not provided, improvements are not tracked,
            early stopping is disabled, and best-model saving is disabled.
        save_settings (optional, SaveSettings):
            Settings for saving the training checkpoint 
            and an optional best model (requires `measure_policy` to be provided).
            If not provided, saving will be disabled.
        perf_settings (optional, PerformanceSettings):
            Settings for training performance.
            If not provided, a default `PerformanceSettings()` instance is used,
            which includes training on CPU.
        log_settings (optional, LogSettings):
            Settings for log formatting.
            If not provided, a default `LogSettings()` instance is used.
    '''
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        targ_key: str,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        sched_settings: Optional[SchedulerSettings] = None,
        eval_settings: Optional[EvalSettings] = None,
        measure_policy: Optional[MeasurePolicy] = None,
        save_settings: Optional[SaveSettings] = None,
        perf_settings: Optional[PerformanceSettings] = None,
        log_settings: Optional[LogSettings] = None
    ):
        self.sched_settings = sched_settings
        self.eval_settings = eval_settings
        self.save_settings = save_settings
        self.perf_settings = PerformanceSettings() if perf_settings is None else perf_settings
        self.log_settings = LogSettings() if log_settings is None else log_settings

        device = self.perf_settings.device # Should already be noramlized to a torch.device
        self.model = model.to(
            device = device,
            memory_format = self.perf_settings.memory_format
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.targ_key = targ_key

        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.measure_policy = measure_policy

        self._validate_config()
        self._print_save_msgs()

        if self.measure_policy is not None:
            self.measure_policy.reset() # Ensure a fresh state

        self.train_history = TrainHistory() # Tracks epoch loss
        self.val_history = ValHistory() # Tracks epoch loss and metrics
        self.start_epoch = 0

        # GradScaler is used for CUDA mixed-precision training (AMP).
        # On other devices, perf_settings.use_amp must be False, so this is a no-op
        self.scaler = torch.amp.GradScaler(
            device = device,
            enabled = self.perf_settings.use_amp
        )

        self.log_sec_div = SEC_DIV_CHAR * self.log_settings.logbox_len
        self.log_end_div = EPOCH_FILL_CHAR * self.log_settings.logbox_len

    def load_checkpoint(self, resume_path: Union[str, Path]) -> None:
        '''
        Loads a training checkpoint into this `ModelTrainer` instance
        from the file at `resume_path`.

        This loads the saved model weights and training state
        (e.g. optimizer, scaler, histories, scheduler, etc.).

        Args:
            resume_path (Union[str, Path]):
                The path to the saved checkpoint file.
        '''
        ckpt_epoch = load_checkpoint(
            checkpoint_path = resume_path,
            model = self.model,
            optimizer = self.optimizer,
            scaler = self.scaler,
            train_history = self.train_history,
            val_history = self.val_history,
            scheduler = self.scheduler,
            measure_policy = self.measure_policy,
            device = self.perf_settings.device
        )
        self.start_epoch = ckpt_epoch + 1
        print(
            f'{BOLD_ON}[NOTE]{BOLD_OFF} '
            f'Successfully loaded checkpoint at {resume_path}. '
            f'Calling self.train(num_epochs) will resume training from epoch {self.start_epoch}.'
        )

    def train(self, num_epochs: int) -> Tuple[TrainHistory, ValHistory]:    
        '''
        Trains the model on the training dataset and evaluates it on the validation dataset.

        Note: See `src.logging.history.TrainHistory` and `src.logging.history.ValHistory`
        for details on the attributes of the returned history objects.

        Args:
            num_epochs (int):
                The maximum number of epochs to train for.
                If early stopping is enabled, training may stop before reaching this limit.
                If resuming from a checkpoint, training continues from `checkpoint_epoch + 1`
                to `num_epochs` (or up to early stopping).

        Returns:
            train_history (TrainHistory):
                Training-phase history containing the epoch losses.
                Can also be accessed from the `train_history` attribute. 
            val_history (ValHistory):
                Validation-phase history containing the epoch losses
                and optional metric values (computed per evaluation interval).
                Can also be accessed from the `val_history` attribute.
        '''    
        for epoch in range(self.start_epoch, num_epochs):
            # ------------------------------------
            # Training step
            # ------------------------------------
            train_start = time.time()
            train_loss = self._train_step()
            train_time = time.time() - train_start

            s_setts = self.sched_settings
            if (s_setts is not None) and (s_setts.step_freq == 'epoch'):
                s_setts.scheduler.step() # Update optimizer learning rates per epoch
          
            # ------------------------------------
            # Validation step
            # ------------------------------------
            val_start = time.time()
            e_setts = self.eval_settings
            should_eval = (
                (e_setts is not None)
                and (epoch % e_setts.eval_interval == 0) 
                and (epoch != 0) # Skip evaluation on zeroth epoch
            )
            val_results = self._val_step(
                metrics = e_setts.metrics if should_eval else None
            )
            val_time = time.time() - val_start

            # ------------------------------------
            # Logging and checkpoint saving
            # ------------------------------------
            self._log_epoch(epoch, train_loss, val_results, train_time, val_time)

            if self.should_save_ckpt:
                save_checkpoint(
                    model = self.model,
                    optimizer = self.optimizer,
                    scaler = self.scaler,
                    train_history = self.train_history,
                    val_history = self.val_history,
                    scheduler = self.scheduler,
                    measure_policy = self.measure_policy,
                    checkpoint_epoch = epoch,
                    save_path = self.save_settings.ckpt_path
                )

            # ------------------------------------
            # Improvement and early stopping
            # ------------------------------------
            if self.measure_policy is None:
                continue

            if self.measure_policy.measure_type == 'loss':
                # Measure policy gets updated every epoch
                improved, should_stop = self.measure_policy.step(val_results['loss'])
            elif should_eval:
                # measure_type='metric', and evaluations were performed
                # Measure policy gets updated every eval interval here
                improved, should_stop = self.measure_policy.step(val_results['metrics'])
            else:
                # measure_type='metric', and no evaluations were performed
                improved, should_stop = False, False

            # Improvement actions
            if improved:
                self.val_history.set_best(
                    value = self.measure_policy.best_score,
                    epoch = epoch,
                    measure_info = self.measure_policy.measure_info
                )

                # Save best model
                if self.should_save_best_model:
                    torch.save(self.model.state_dict(), self.save_settings.best_model_path)

            # Early stopping check
            if should_stop:
                print(
                    f'{BOLD_ON}[NOTE]{BOLD_OFF} No improvement detected in measure policy after '
                    f'{self.measure_policy.no_improve_counter} steps. Early stopping triggered.'
                )
                return self.train_history, self.val_history

        return self.train_history, self.val_history
           
    def _train_step(self) -> torch.Tensor:
        '''
        Performs the training phase of a single epoch.
        This loops through the training dataset once, computing the loss 
        for each batch and updating the model weights.

        Note: The epoch loss is computed as the mean of batch losses across the epoch.
              Consequently, it may not equal the mean loss across all samples.

        Returns:
            torch.Tensor:
                Epoch loss (mean of batch losses across the epoch).
        '''
        num_batches = len(self.train_loader)
        loss_sum = 0

        self.model.train()
        for batch in self.train_loader:
            p_setts = self.perf_settings
            device = p_setts.device
            scaler = self.scaler

            imgs = batch['image'].to(device, memory_format = p_setts.memory_format)
            targs = batch[self.targ_key].to(device)
            
            self.optimizer.zero_grad() # Zero parameter gradients

            # The AMP context is only relevant for CUDA. 
            # It is disabled for other devices (p_setts.use_amp = False)
            with torch.autocast(
                device_type = device.type, 
                dtype = p_setts.amp_dtype, 
                enabled = p_setts.use_amp
            ):
                # Compute batch loss
                logits = self.model(imgs)
                loss = self.loss_fn(logits, targs)

            scaler.scale(loss).backward() # Backpropagate on batch loss
            scaler.step(self.optimizer) # Update parameters
            scaler.update() # Update grad scalar

            s_setts = self.sched_settings
            if (s_setts is not None) and (s_setts.step_freq == 'optim_step'):
                s_setts.scheduler.step() # Update learning rates per optimizer step
                
            loss_sum += loss.detach() # Update running sum loss

        return loss_sum / num_batches # Epoch loss (mean of batch losses)

    def _val_step(self, metrics: Optional[Dict[str, Metric]]) -> ValResults:
        '''
        Performs the validation phase of a single epoch.
        This loops through the validation dataset once, computing the loss 
        for each batch and optionally updating the metrics. 
        The metrics are computed after all batches have been processed.

        Note: See `src.metrics.types` for an example structure of `MetricResults`.

        Note: The epoch loss is computed as the mean of batch losses across the epoch.
              Consequently, it may not equal the mean loss across all samples.

        Args:
            metrics (optional, Dict[str, Metric]):
                Dictionary mapping task names to metric objects (must implement the `Metric` protocol).
                Each of the metric objects are reset before processing the batches.
        Returns:
            ValResults:
                Dictionary containing validation results for the epoch.
                This includes:
                    - loss (torch.Tensor): 
                        Epoch loss (mean of batch losses across the epoch).
                    - metrics (optional, MetricResults):
                        Dictionary of computed metric values.
                        This is `None` if `metrics` was not provided as input.
        '''
        num_batches = len(self.val_loader)
        loss_sum = 0

        # Reset evaluation metrics
        if metrics is not None:
            for metric in metrics.values():
                metric.reset()

        # Start step loop  
        self.model.eval()
        for batch in self.val_loader:
            p_setts = self.perf_settings
            device = p_setts.device

            imgs = batch['image'].to(device, memory_format = p_setts.memory_format)
            targs = batch[self.targ_key].to(device)

            with torch.inference_mode():
                # The AMP context is only relevant for CUDA. 
                # It is disabled for other devices (p_setts.use_amp = False)
                with torch.autocast(
                    device_type = device.type, 
                    dtype = p_setts.amp_dtype, 
                    enabled = p_setts.use_amp
                ):
                    # Compute batch loss
                    logits = self.model(imgs)
                    loss = self.loss_fn(logits, targs)
                
            loss_sum += loss # Update running sum loss
            
            if metrics is not None:
                # Update evaluation metric states
                preds = logits.argmax(dim = 1) # Prediction probabilities
                for metric in metrics.values():
                    metric.update(preds, targs)

        # Compute dataset summary values (loss and evaluation metrics)
        if metrics is not None:
            metric_results = {name: metric.compute() for name, metric in metrics.items()}
        else:
            metric_results = None

        return {
            'loss': loss_sum / num_batches, # Epoch loss (mean of batch losses)
            'metrics': metric_results
        }

    def _log_epoch(
        self, 
        epoch: int, 
        train_loss: torch.Tensor,
        val_results: ValResults,
        train_time: float, 
        val_time: float
    ) -> None:
        '''
        Formats and prints the training and validation logs of an epoch.
        Also records the loss values and metrics into `train_history` and `val_history`.

        Args:
            epoch (int):
                The epoch index being logged.
            train_loss (torch.Tensor):
                Epoch loss from the training phase (`_training_step`).
            val_results (ValResults):
                Results dictionary from the validation phase (`_val_step`).
                This contains:
                    - loss (torch.Tensor): Epoch loss.
                    - metrics (optional, MetricResults): Dictionary of computed metric values.
            train_time (float):
                Computation time for the training phase of the epoch.
            val_time (float):
                Computation time for the validation phase of the epoch.
        '''
        setts = self.log_settings
        log_kwargs = {
            'logbox_len': setts.logbox_len,
            'max_row_entries': setts.max_row_entries,
            'num_decimals': setts.num_decimals
        }

        # List to store all logging sections
        epoch_log_secs = [
            make_epoch_header(epoch, setts.logbox_len), 
            self.log_sec_div
        ]

        # Record loss and evalutation metrics
        rec_train_loss, _ = self.train_history.record(loss_value = train_loss, epoch = epoch)
        rec_val_loss, rec_val_metrics = self.val_history.record(
            loss_value = val_results['loss'], 
            metric_results = val_results['metrics'],
            epoch = epoch
        )

        # Make loss log section for both training and validation
        loss_log_sec = make_log_sec(
            sec_name = 'loss',
            entry_names = ['train (mean)', 'val (mean)'],
            entry_values = [rec_train_loss, rec_val_loss],
            **log_kwargs
        )
        epoch_log_secs.extend([loss_log_sec, self.log_sec_div])

        # Make evaluation metric log section for validation
        if (rec_val_metrics is not None) and (self.should_log_metrics):
            metric_log_sec = make_metric_log_sec(
                metric_results = rec_val_metrics,
                metric_specs = self.eval_settings.log_metric_specs,
                **log_kwargs
            )
            epoch_log_secs.extend([metric_log_sec, self.log_end_div])

        # Make time log section
        time_log_sec = make_log_sec(
            sec_name = 'time',
            entry_names = ['train', 'val'],
            entry_values = [train_time, val_time],
            entry_units = 'sec',
            **log_kwargs
        )      
        epoch_log_secs.extend([time_log_sec, self.log_end_div, ''])

        # Print all log sections
        for log_str in epoch_log_secs:
            print(log_str)

    def _validate_config(self) -> None:   
        '''
        Validates the configuration of the model trainer.
        '''    
        # Check if eval_settings is provided when measure_info = 'metric'
        mp = self.measure_policy
        metric_policy_no_eval = (
            (mp is not None)
            and (mp.measure_info == 'metric')
            and (self.eval_settings is None)
        )
        if metric_policy_no_eval:
            warnings.warn(
                "measure_policy.measure_info = 'metric', but no eval_settings was provided."
                'measure_policy will be ignored.'
            )
        
    def _print_save_msgs(self) -> None:
        '''
        Prints information about where the training checkpoint and the best model will be saved.
        Prints warnings if checkpoint or best-model saving is disabled.
        '''
        setts = self.save_settings
        if not self.should_save_ckpt:
            warnings.warn(
                f'No checkpoint save path provided. Training checkpoint will not be saved.',
                UserWarning
            )
        else:
            print(f'{BOLD_ON}[NOTE]{BOLD_OFF} Training checkpoint will be saved at: {setts.ckpt_path}')

        if self.measure_policy is not None:
            if not self.should_save_best_model:
                warnings.warn(
                    f'No best-model save path provided, but a measure policy was provided.'
                    'The best model will not be saved.',
                    UserWarning
                )
            else:
                print(f'{BOLD_ON}[NOTE]{BOLD_OFF} Best model will be saved at: {setts.best_model_path}')
        print() # Adds an extra spacing at the end

    @property
    def should_save_best_model(self)-> bool:
        '''
        Best-model saving is available.

        Note: During training, saving a best model is only ever considered if
              `measure_policy` was provided at initialization.
        '''
        setts = self.save_settings
        return (setts is not None) and (setts.best_model_name is not None)

    @property
    def should_save_ckpt(self) -> bool:
        '''
        Whether checkpoint saving is available.
        '''
        setts = self.save_settings
        return (setts is not None) and (setts.ckpt_name is not None)

    @property
    def should_log_metrics(self) -> bool:
        '''
        Whether logging evaluation metrics is available.
        '''
        setts = self.eval_settings
        return(setts is not None) and (setts.log_metric_specs is not None)
    
    @property
    def scheduler(self) -> Optional[LRScheduler]:
        '''
        Learning rate scheduler.
        Returns `None` if `sched_settings` was not provided at initialization.
        '''
        if self.sched_settings is None:
            return None
        else:
            return self.sched_settings.scheduler
        
    @property
    def metrics(self) -> Optional[Dict[str, Metric]]:
        '''
        Metrics used to evaluate the model on the validation dataset.
        Returns `None` if `eval_settings` was not provided at initialization.
        '''
        if self.eval_settings is None:
            return None
        else:
            return self.eval_settings.metrics