#####################################
# Imports & Dependencies
#####################################
from torch import nn
from pathlib import Path
from typing import Union, Dict, Any, Optional

from src.models.modules import ACTIVATIONS
from src.utils.file_utils import load_yaml_config

SUPPORTED_ACTIVATIONS = ', '.join(ACTIVATIONS.keys())


#####################################
# Functions
#####################################
def get_activation(key: str, context: str = 'activation') -> nn.Module:
    '''
    Instantiates an activation module from the `ACTIVATIONS` dictionary.

    Args:
        key (str):
            Name of the activation function to instantiate.
            This must be a key in `ACTIVATIONS` (e.g. `'relu', 'gelu', 'sigmoid'`)
        context (str):
            Label used to describe what the activation is used for.
            This is to provide more specific error messages.
            Default is `activation`.
    Returns:
        nn.Module:
            Instantiated activation module.
    '''
    try:
        activation = ACTIVATIONS[key]()
    except KeyError:
        raise KeyError(
            f"Unknown {context}: {key}"
            f"Supported activations are {SUPPORTED_ACTIVATIONS}"
        )
    return activation


def load_mit_config(config_file: Union[str, Path], in_channels: Optional[int]) -> Dict[str, Any]:
    '''
    Constructs a `MixTransformer` config dictionary from loading a YAML config file.

    Notes on In-Channels:
        - `in_channels` is considered a task-specific argument.
          
        - If `in_channels` is provided,
          its value overrides any corresponding value contained in `config_file`.
          
        - If `in_channels` is not provided, 
          its value needs to be contained in `config_file`
          or passed directly to `MixTransformer` when instantiating.

    Args:
        config_file (Union[str, Path]):
            Path to a YAML config file containing all 
            required arguments for `MixTransformer`.
            If the config includes `enc_activations`,
            their values must be represented by activation names/keys in `ACTIVATIONS`.
            (e.g. `'relu', 'gelu', 'sigmoid'`).
        in_channels (optional, int):
            Number of input channels.
            If provided, this value overrides any 'in_channels' value in `config_file`.
            
    Returns:
        Dict[str, Any]:
            Config dictionary used to instantiate a `MixTransformer`.

    Examples:
        Create a `MixTransformer` model from a config file:
        
        >>> config = load_mit_config(config_file)
        >>> mit = MixTransformer(**config)
    '''
    config = load_yaml_config(config_file)

    # Override task-specific arguments if necessary
    if in_channels is not None:
        config['in_channels'] = in_channels

    enc_activation = config.get('enc_activations', None)
    if enc_activation is None:
        return config
    
    if isinstance(enc_activation, list):
        config['enc_activations'] = [
            get_activation(key, 'encoder activation') 
            for key in enc_activation
        ]
    else:
        config['enc_activations'] = get_activation(enc_activation, 'encoder activation')
        
    return config


def load_segformer_config(
    config_file: Union[str, Path], 
    in_channels: Optional[int], 
    num_classes: Optional[int]
) -> Dict[str, Any]:
    '''
    Constructs a `SegFormer` config dictionary from loading a YAML config file.

    Notes on Task-Specific Arguments:
        - The task-specific arguments are `in_channels` and `num_classes`.
          
        - If either `in_channels` or `num_classes` are provided,
          their values override any corresponding value contained in `config_file`.
          
        - If either `in_channels` or `num_classes` are not provided, 
          their values need to be contained in `config_file`
          or passed directly to `SegFormer` when instantiating.

    Args:
        config_file (Union[str, Path]):
            Path to a YAML config file containing the arguments for `SegFormer`.
            If `enc_activations` or `dec_activations` are included,
            their values must be represented by activation names/keys in `ACTIVATIONS`.
            (e.g. `'relu', 'gelu', 'sigmoid'`).
        in_channels (optional, int):
            Number of input channels in the SegFormer encoder.
            If provided, this value overrides any 'in_channels' value in `config_file`.
        num_classes (optional, int):
            Number of classes predicted by the SegFormer decoder.
            If provided, this value overrides any 'num_classes' value in `config_file`.
    Returns:
        Dict[str, Any]:
            Config dictionary used to instantiate a `SegFormer`.

    Examples:
        Create a `SegFormer` model directly from a config file:

        >>> config = load_segformer_config(config_file)
        >>> segformer = SegFormer(**config)
    '''
    config = load_yaml_config(config_file)

    # Override task-specific arguments if necessary
    if in_channels is not None:
        config['in_channels'] = in_channels
        
    if num_classes is not None:
        config['num_classes'] = num_classes

    # Get encoder activation function
    enc_activation = config.get('enc_activations')
    if enc_activation is None:
        pass
    elif isinstance(enc_activation, list):
        config['enc_activations'] = [
            get_activation(key, 'encoder activation') 
            for key in enc_activation
        ]
    else:
        config['enc_activations'] = get_activation(enc_activation, 'encoder activation')
        
    # Get decoder activation function
    dec_activation = config.get('dec_activation')
    if dec_activation is not None:
        config['dec_activation'] = get_activation(dec_activation, 'decoder activation')
    
    return config


    

    