#####################################
# Imports & Dependencies
#####################################
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler

import time
from pathlib import Path
import warnings
from typing import Optional, Union, Dict, Tuple, TypedDict

from src.logging.history import TrainHistory, ValHistory
from src.logging.formatting import (
    BOLD_ON, BOLD_OFF, EPOCH_FILL_CHAR, SEC_DIV_CHAR,
    make_epoch_header, make_metric_log_sec, make_log_sec
)

from src.metrics.ops import Metric
from src.metrics.types import MetricResults
from src.engine.checkpoint import load_checkpoint, save_checkpoint
from src.engine.trainer_configs import (
    SchedulerConfig, EvalConfig, 
    SaveConfig, PerformanceConfig, LogConfig
)
from src.engine.measure_policy import MeasurePolicy


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
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        targ_key: str,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler_cfg: Optional[SchedulerConfig] = None,
        eval_cfg: Optional[EvalConfig] = None,
        measure_policy: Optional[MeasurePolicy] = None,
        save_cfg: Optional[SaveConfig] = None,
        log_cfg: Optional[LogConfig] = None,
        perf_cfg: Optional[PerformanceConfig] = None
    ):
        self.scheduler_cfg = scheduler_cfg
        self.eval_cfg = eval_cfg
        self.save_cfg = save_cfg
        self.perf_cfg = PerformanceConfig() if perf_cfg is None else perf_cfg
        self.log_cfg = LogConfig() if log_cfg is None else log_cfg

        device = self.perf_cfg.device # Should already be noramlized to a torch.device
        self.model = model.to(
            device = device,
            memory_format = self.perf_cfg.memory_format
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

        self.train_history = TrainHistory() # Tracks average batch losses
        self.val_history = ValHistory() # Tracks average batch losses and metrics
        self.start_epoch = 0

        # GradScaler is used for CUDA mixed-precision training (AMP).
        # On other devices, perf_cfg.use_amp must be False, so this is a no-op
        self.scaler = torch.amp.GradScaler(
            device = device,
            enabled = self.perf_cfg.use_amp
        )

        self.log_sec_div = SEC_DIV_CHAR * self.log_cfg.logbox_len
        self.log_end_div = EPOCH_FILL_CHAR * self.log_cfg.logbox_len

    def load_checkpoint(self, resume_path: Union[str, Path]) -> None:
        ckpt_epoch = load_checkpoint(
            checkpoint_path = resume_path,
            model = self.model,
            optimizer = self.optimizer,
            scaler = self.scaler,
            train_history = self.train_history,
            val_history = self.val_history,
            scheduler = self.scheduler,
            measure_policy = self.measure_policy,
            device = self.device
        )
        self.start_epoch = ckpt_epoch + 1
        print(
            f'{BOLD_ON}[NOTE]{BOLD_OFF} '
            f'Successfully loaded checkpoint at {resume_path}. '
            f'Calling self.train(num_epochs) will resume training from epoch {self.start_epoch}.'
        )

    def train(self, num_epochs: int) -> Tuple[TrainHistory, ValHistory]:        
        for epoch in range(self.start_epoch, num_epochs):
            # ------------------------------------
            # Training step
            # ------------------------------------
            train_start = time.time()
            train_loss = self._train_step()
            train_time = time.time() - train_start

            cfg = self.scheduler_cfg
            if (cfg is not None) and (cfg.step_freq == 'epoch'):
                cfg.scheduler.step() # Update optimizer learning rates per epoch
          
            # ------------------------------------
            # Validation step
            # ------------------------------------
            val_start = time.time()
            cfg = self.eval_cfg
            should_eval = (
                (cfg is not None)
                and (epoch % cfg.eval_interval == 0) 
                and (epoch != 0) # Skip evaluation on zeroth epoch
            )
            val_results = self._val_step(
                metrics = cfg.metrics if should_eval else None
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
                    save_path = self.save_cfg.ckpt_path
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
                    torch.save(self.model.state_dict(), self.save_cfg.best_model_path)

            # Early stopping check
            if should_stop:
                print(
                    f'{BOLD_ON}[NOTE]{BOLD_OFF} No improvement detected in measure policy after '
                    f'{self.measure_policy.no_improve_counter} steps. Early stopping triggered.'
                )
                return self.train_history, self.val_history

        return self.train_history, self.val_history
           
    def _train_step(self) -> torch.Tensor:
        num_batches = len(self.train_loader)
        loss_sum = 0

        self.model.train()
        for batch in self.train_loader:
            pcfg = self.perf_cfg
            device = pcfg.device
            scaler = self.scaler

            imgs = batch['image'].to(device, memory_format = pcfg.memory_format)
            targs = batch[self.targ_key].to(device)
            
            self.optimizer.zero_grad() # Zero parameter gradients

            # The AMP context is only relevant for CUDA. 
            # It is disabled for other devices (pcfg.use_amp = False)
            with torch.autocast(
                device_type = device.type, 
                dtype = pcfg.amp_dtype, 
                enabled = pcfg.use_amp
            ):
                # Compute loss (sum and average) for batch
                logits = self.model(imgs)
                loss = self.loss_fn(logits, targs)

            scaler.scale(loss).backward() # Backpropagate on average batch loss
            scaler.step(self.optimizer) # Update parameters
            scaler.update() # Update grad scalar

            scfg = self.scheduler_cfg
            if (scfg is not None) and (scfg.step_freq == 'optim_step'):
                scfg.scheduler.step() # Update learning rates per optimizer step
                
            loss_sum += loss.detach() # Update running sum loss

        return loss_sum / num_batches # Average batch loss

    def _val_step(self, metrics: Optional[Dict[str, Metric]]) -> ValResults:
        num_batches = len(self.val_loader)
        loss_sum = 0

        # Reset evaluation metrics
        if metrics is not None:
            for metric in metrics.values():
                metric.reset()

        # Start step loop  
        self.model.eval()
        for batch in self.val_loader:
            pcfg = self.perf_cfg
            device = pcfg.device

            imgs = batch['image'].to(device, memory_format = pcfg.memory_format)
            targs = batch[self.targ_key].to(device)

            with torch.inference_mode():
                # The AMP context is only relevant for CUDA. 
                # It is disabled for other devices (pcfg.use_amp = False)
                with torch.autocast(
                    device_type = device.type, 
                    dtype = pcfg.amp_dtype, 
                    enabled = pcfg.use_amp
                ):
                    # Compute loss (sum and average) for batch
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
            'loss': loss_sum / num_batches, # Average batch loss
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
        log_kwargs = {
            'logbox_len': self.log_cfg.logbox_len,
            'max_row_entries': self.log_cfg.max_row_entries,
            'num_decimals': self.log_cfg.num_decimals
        }

        # List to store all logging sections
        epoch_log_secs = [
            make_epoch_header(epoch, self.log_cfg.logbox_len), 
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
                metric_specs = self.eval_cfg.log_metric_specs,
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
        # Check if eval_cfg is provided when measure_info = 'metric'
        mp = self.measure_policy
        metric_policy_no_eval = (
            (mp is not None)
            and (mp.measure_info == 'metric')
            and (self.eval_cfg is None)
        )
        if metric_policy_no_eval:
            warnings.warn(
                "measure_policy.measure_info = 'metric', but no eval_cfg was provided."
                'measure_policy will be ignored.'
            )
        
    def _print_save_msgs(self) -> None:
        cfg = self.save_cfg
        if not self.should_save_ckpt:
            warnings.warn(
                f'ckpt_name not provided in save_cfg. Training checkpoints will not be saved.',
                UserWarning
            )
        else:
            print(f'{BOLD_ON}[NOTE]{BOLD_OFF} Checkpoints will be saved at: {cfg.ckpt_path}')

        if self.measure_policy is not None:
            if not self.should_save_best_model:
                warnings.warn(
                    f'best_model_name not provided in save_cfg, but measure_policy was provided.'
                    'The best model will not be saved.',
                    UserWarning
                )
            else:
                print(f'{BOLD_ON}[NOTE]{BOLD_OFF} Best model will be saved at: {cfg.best_model_path}')
        print() # Adds an extra spacing at the end

    @property
    def should_save_best_model(self)-> bool:
        '''
        Returns whether best model saving is available.
        Saving a best model is only considered when 
        `self.measure_policy` is provided to determine improvement.
        '''
        cfg = self.save_cfg
        return (cfg is not None) and (cfg.best_model_name is not None)

    @property
    def should_save_ckpt(self) -> bool:
        '''
        Returns whether checkpoint saving is available.
        '''
        cfg = self.save_cfg
        return (cfg is not None) and (cfg.ckpt_name is not None)

    @property
    def should_log_metrics(self) -> bool:
        cfg = self.eval_cfg
        return(cfg is not None) and (cfg.log_metric_specs is not None)
    
    @property
    def scheduler(self) -> Optional[lr_scheduler._LRScheduler]:
        if self.scheduler_cfg is None:
            return None
        else:
            return self.scheduler_cfg.scheduler
        
    @property
    def metrics(self) -> Optional[Dict[str, Metric]]:
        if self.eval_cfg is None:
            return None
        else:
            return self.eval_cfg.metrics