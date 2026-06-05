#####################################
# Imports & Dependencies
#####################################
from torch import nn
import yaml
from pathlib import Path
from typing import Union, Dict, Any

from src.models.modules import ACTIVATIONS

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


def load_mit_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    '''
    Constructs a `MixTransformer` config dictionary from loading a YAML config file.

    Args:
        config_file (Union[str, Path]):
            Path to a YAML config file containing all 
            required arguments for `MixTransformer`.
            If the config includes `enc_activations`,
            their values must be represented by activation names/keys in `ACTIVATIONS`.
            (e.g. `'relu', 'gelu', 'sigmoid'`).

    Returns:
        Dict[str, Any]:
            Config dictionary that can be directly used to instantiate a `MixTransformer`.

    Examples:
        Create a `MixTransformer` model from a config file:
        
        >>> config = load_mit_config(config_file)
        >>> mit = MixTransformer(**config)
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
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


def load_segformer_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    '''
    Constructs a `SegFormer` config dictionary from loading a YAML config file.

    Args:
        config_file (Union[str, Path]):
            Path to a YAML config file containing all 
            required arguments for `SegFormer`.
            If the config includes `enc_activations` or `dec_activations`,
            their values must be represented by activation names/keys in `ACTIVATIONS`.
            (e.g. `'relu', 'gelu', 'sigmoid'`).
    Returns:
        Dict[str, Any]:
            Config dictionary that can be directly used to instantiate a `SegFormer`.

    Examples:
        Create a `SegFormer` model from a config file:

        >>> config = load_segformer_config(config_file)
        >>> segformer = SegFormer(**config)
    '''
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

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


    

    