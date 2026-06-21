'''
Script to train a `SegFormer` model 
on a semantic segmentation dataset (e.g. filtered Supervisely Person).

Example usage:
    `python run_training.py config.yaml`
'''


# #####################################
# # Imports & Dependencies
# #####################################
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim import AdamW

from argparse import ArgumentParser
import warnings
warnings.filterwarnings('ignore', category = UserWarning)

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[2] # Script is in segformer-pytorch/dir_1/dir_2/
sys.path.append(str(repo_root))

from src.models.segformer import SegFormer
from src.models.config_loaders import load_segformer_config

from src.data_setup.datasets import SuperviselyPersonFiltered
from src.data_setup.dataloader_utils import build_dataloader
from src.data_setup.transforms.pipelines import get_phot_transforms, get_geo_transforms

from src.losses import FocalDiceLoss
from src.metrics.ops import SegmentationMetrics
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
    parser = ArgumentParser(description = 'Train SegFormer model on the filtered Supervisely Person dataset.')
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
    train_tf, val_tf = {}, {}
    if tf_cfg is not None:
        # Photometric transforms
        if tf_cfg.get('include_train_phot'):
            train_tf['img_phot_transforms'] = get_phot_transforms()

        if tf_cfg.get('include_val_phot'):
            val_tf['img_phot_transforms'] = get_phot_transforms()

        # Geometric transforms
        common_geo_cfg = tf_cfg.get('common_geo') or {}
        train_geo_cfg = tf_cfg.get('train_geo') or {}
        val_geo_cfg = tf_cfg.get('val_geo') or {}
        
        train_geo_cfg = common_geo_cfg | train_geo_cfg
        val_geo_cfg = common_geo_cfg | val_geo_cfg

        if train_geo_cfg:
            train_tf['geo_transforms'] = get_geo_transforms(**train_geo_cfg)
        
        if val_geo_cfg:
            val_tf['geo_transforms'] = get_geo_transforms(**val_geo_cfg)


    # -----------------------------------
    # Dataset and Dataloaders
    # -----------------------------------
    # Datasets (Human-NonHuman)
    dataset_cfg = cfg['dataset']
    dataset_info = dataset_cfg['info']
    dataset_args = dataset_cfg['args']
    dataset_args['root'] = resolve_path(dataset_args['root'], cfg_dir)

    train_dataset = SuperviselyPersonFiltered(
        split = 'train',
        **train_tf,
        **dataset_args
    )
    val_dataset = SuperviselyPersonFiltered(
        split = 'val',
        **val_tf,
        **dataset_args
    )

    ignore_idx = train_dataset.ignore_idx

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
    # SegFormer model
    model_cfg = cfg['model']
    segformer_cfg = load_segformer_config(
        resolve_path(model_cfg['segformer_config_file'], cfg_dir),
        in_channels = dataset_info['in_channels'],
        num_classes = dataset_info['num_classes']
    )
    model = SegFormer(**segformer_cfg)

    # Load MiT backbone weights
    if model_cfg.get('mit_weights'):
        mit_state_dict = torch.load(
            resolve_path(model_cfg['mit_weights'], cfg_dir),
            map_location = device
        )
        model.encoder.load_state_dict(mit_state_dict)


    # -----------------------------------
    # Loss and Optimizer
    # -----------------------------------
    # Loss Function (Focal-Dice)
    loss_cfg = cfg['loss']
    if ignore_idx is not None:
        loss_cfg['ignore_idx'] = ignore_idx

    loss_fn = FocalDiceLoss(**loss_cfg)

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

    # Evaluation Settings (SegmentationMetrics)
    if cfg.get('eval_settings') is not None:
        metrics = {
            'seg': SegmentationMetrics(dataset_info['num_classes'], ignore_idx)
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