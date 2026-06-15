'''
Script to train a `MiTClassification` model 
on an image classification dataset (e.g. Mini-ImageNet).

Example usage:
    `python run_training.py config.yaml`
'''


# #####################################
# # Imports & Dependencies
# #####################################
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

from argparse import ArgumentParser

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2] # Script is in segformer-pytorch/dir_1/dir_2/
sys.path.append(str(repo_root))

from src.models import MiTClassification
from src.models.config_loaders import load_mit_config

from src.data_setup.datasets import MiniImageNetDataset
from src.data_setup.dataloader_utils import build_dataloader
from src.data_setup.transforms.pipelines import get_transforms

from src.metrics import ClassificationMetrics
from src.engine.measure_policy import MeasurePolicy

from src.engine.trainer import ModelTrainer
from src.engine.trainer_settings import EvalSettings, SaveSettings, LogSettings

from src.engine.settings_builders import make_perf_settings, make_sched_settings
from src.utils.ml_utils import set_seed, get_device
from src.utils.file_utils import load_yaml_config, resolve_path


#####################################
# Training Code
#####################################
if __name__ == '__main__':
    # Setup parser and load config
    parser = ArgumentParser(description = 'Train MiTClassification model on the Mini-ImageNet dataset.')
    parser.add_argument(
        'config_file', 
        help = 'Path to the YAML configuration file.',
        type = str
    )
    parser.add_argument(
        '-s', '--seed', 
        help = 'Set random seed.',
        type = int, 
        default = 0
    )
    
    args = parser.parse_args()
    cfg_dir = Path(args.config_file).resolve().parent
    
    cfg = load_yaml_config(args.config_file) # Load training cfg file
    set_seed(args.seed) # Set seed for reproducibility
    device = cfg['device'] if cfg.get('device') is not None else get_device()


    # -----------------------------------
    # Transforms
    # -----------------------------------
    tf_cfg = cfg.get('transforms')
    train_tf, val_tf = None, None
    if tf_cfg is not None:
        common_args = tf_cfg.get('common_cfg') or {}
        train_tf_cfg = tf_cfg.get('train') or {}
        val_tf_cfg = tf_cfg.get('val') or {}

        train_tf_cfg = common_args | train_tf_cfg
        val_tf_cfg = common_args | val_tf_cfg
        
        if train_tf_cfg:
            train_tf = get_transforms(**train_tf_cfg)
        
        if val_tf_cfg:
            val_tf = get_transforms(**val_tf_cfg)


    # -----------------------------------
    # Dataset and Dataloaders
    # -----------------------------------
    # Datasets (Mini-ImageNet)
    dataset_cfg = cfg['dataset']
    dataset_info = dataset_cfg['info']
    dataset_args = dataset_cfg['args']
    dataset_args['root'] = resolve_path(dataset_args['root'], cfg_dir)

    train_dataset = MiniImageNetDataset(
        split = 'train',
        transforms = train_tf,
        **dataset_args
    )
    val_dataset = MiniImageNetDataset(
        split = 'val',
        transforms = val_tf,
        **dataset_args
    )

    # Dataloaders
    loader_cfg = cfg['dataloader']
    loader_cfg['device'] = device

    train_loader = build_dataloader(
        dataset = train_dataset,
        split = 'train',
        **loader_cfg
    )
    val_loader = build_dataloader(
        dataset = val_dataset,
        split = 'val',
        **loader_cfg
    )


    # -----------------------------------
    # Model
    # -----------------------------------
    # MiTClassification model
    mit_cfg = load_mit_config(
        config_file = resolve_path(cfg['mit_config_file'], cfg_dir),
        in_channels = dataset_info['in_channels']
    )
    model = MiTClassification.from_mit_config(
        mit_config = mit_cfg,
        num_classes = dataset_info['num_classes']
    )


    # -----------------------------------
    # Loss and Optimizer
    # -----------------------------------
    # Loss Function (Cross Entropy)
    loss_fn = nn.CrossEntropyLoss(**cfg['loss'])

    # Optimizer (AdamW)
    optimizer = AdamW(model.parameters(), **cfg['optimizer'])


    # -----------------------------------
    # Optional Settings
    # -----------------------------------
    trainer_specs = {} # Stores settings and measure policy

    # Scheduler Settings (CosineAnnealingLR)
    if cfg.get('sched_settings') is not None:
        trainer_specs['sched_settings'] = make_sched_settings(
            optimizer,
            sched_class = CosineAnnealingLR,
            num_optim_steps = len(train_dataset),
            **cfg['sched_settings']
        )

    # Evaluation Settings (ClassificationMetrics)
    if cfg.get('eval_settings') is not None:
        metrics = {
            'cls': ClassificationMetrics(num_classes = dataset_info['num_classes'])
        }
        trainer_specs['eval_settings'] = EvalSettings(metrics, **cfg['eval_settings'])

    # Measure Policy
    if cfg.get('measure_policy') is not None:
        trainer_specs['measure_policy'] = MeasurePolicy(**cfg['measure_policy'])

    # Save Settings
    if cfg.get('save_settings') is not None:
        save_cfg = cfg['save_settings']
        save_cfg['save_dir'] = resolve_path(save_cfg['save_dir'], cfg_dir)
        trainer_specs['save_settings'] = SaveSettings(**save_cfg)

    # Performance Settings
    perf_cfg = cfg.get('perf_settings')
    if perf_cfg is not None:
        perf_cfg['device'] = device
        trainer_specs['perf_settings'] = make_perf_settings(perf_cfg)

    # Log Settings
    if cfg.get('log_settings') is not None:
        trainer_specs['log_settings'] = LogSettings(**cfg['log_settings'])


    # -----------------------------------
    # Model Trainer
    # -----------------------------------
    trainer = ModelTrainer(
        model = model,
        train_loader = train_loader,
        val_loader = val_loader,
        targ_key = dataset_info['targ_key'],
        loss_fn = loss_fn,
        optimizer = optimizer,
        **trainer_specs
    )
    
    
    # -----------------------------------
    # Training
    # -----------------------------------
    # Resume if provided
    if cfg.get('resume_path') is not None:
        resume_path = resolve_path(cfg['resume_path'], cfg_dir)
        trainer.load_checkpoint(resume_path)

    # Start training
    _, _ = trainer.train(cfg['num_epochs'])

    # Save backbone weights
    if 'save_settings' in trainer_specs:
        save_dir = trainer_specs['save_settings'].save_dir
        model.save_mit_backbone(
            save_path = save_dir / 'best_backbone.pth'
        )