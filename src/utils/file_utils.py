#####################################
# Imports & Dependencies
#####################################
import yaml
from pathlib import Path

from typing import Union, Any, Optional, Dict


#####################################
# Functions
#####################################
def format_file_path(file_path: Union[str, Path], path_name: Optional[str] = None) -> Path:
    '''
    Formats and validate a file path.
    This converts all `path` inputs into `pathlib.Path` objects.
    It also checks that `path` contains a file extension 
    and does not end with a path separator ('/' or '\\').

    Args:
        file_path (Union[str, Path]): 
            The path to format and validate.
        path_name (optional, str): 
            Name of `file_path` to use for error messages.

    Returns:
        Path: 
            The validated `pathlib.Path` object.
    '''
    path_name = 'path' if path_name is None else path_name

    path = Path(file_path)
    if str(path).endswith(('/', '\\')):
        raise ValueError(
            f"{path_name} must not end with a path separator ('/' or '\\'). Got: {file_path}"
        )
    
    if path.suffix == '':
        raise ValueError(
            f'{path_name} must end with a file extension. Got: {file_path}'
        )
    
    return path


def load_yaml_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    '''
    Checks for the existence of a YAML config file and 
    loads it into a dictionary if it is non-empty.

    Args:
        config_file (Union[str, Path]):
            Path to the config file.

    Returns:
        Dict[str, Any]:
            Config dictionary from loading the config file.
    '''
    config_file = format_file_path(config_file, path_name = 'config file')

    if not config_file.is_file():
        raise FileNotFoundError(f'Config file not found: {config_file}')

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError('Config file is empty.')

    return config