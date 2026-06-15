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


def resolve_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    '''
    Resolves a file/directory path to an absolute path.

    If `path` is absolute, it is returned as a `Path` object.
    If `path` is relative, it is resolved to an absolute path 
    using the following assumptions:
        - If `base_dir` is provided, `path` is assumed relative to `base_dir`.
        - If `base_dir` is not provided, `path` is assumed relative to the current working directory.

    Args:
        path (Union[str, Path]):
            The path to resolve into an absolute path.
        base_dir (optional, Union[str, Path]):
            Base directory used as a reference point when `path` is a relative path.
            If not provided, the reference point is set to the current working directory.

    Returns:
        Path: 
            The resolved absolute path.
    '''
    path = Path(path)

    if path.is_absolute() or base_dir is None:
        return path.resolve()
    else:
        return (Path(base_dir) / path).resolve()



# def find_repo_root(start_path: Union[str, Path]) -> Path:
#     '''
#     Finds the root directory of a Git repository
#     given a starting path within the repository.

#     This is done by finding the parent directory which contains the '.git' directory.

#     Args:
#         start_path: Union[str, Path]: The file/directory path within the repository.

#     Returns:
#         Path: The absolute path to the root directory of the repository.
#     '''
#     start_path = Path(start_path).resolve() # Normalize to path object
    
#     # Loop through parents and search for .git
#     for p in [start_path, *start_path.parents]:
#         if (p / '.git').exists():
#             return p
        
#     raise ValueError(
#         "Could not find repo root. "
#         "Please ensure you are inside the repo and "
#         "that the repo contains a '.git' directory."
#     )