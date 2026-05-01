#####################################
# Imports & Dependencies
#####################################
import torch
import math
import warnings

from typing import Optional, Sequence

from src.ml_types import MetricLogFields, EntryLogUnits
from src.utils import nested_extract, apply_agg
from src.logging.history import HistoryResults

BOLD_ON = '\033[1m'
BOLD_OFF = '\033[0m'
EPOCH_FILL_CHAR = '='
SEC_DIV_CHAR = '-'
ROW_DIV_CHAR = '|'


#####################################
# Functions
#####################################
def make_epoch_header(epoch: int, logbox_len: int = 100, epoch_width: int = 3) -> str:
    '''
    Creates the string used to indicate the epoch index during logging.
    '''    
    epoch_str = f' EPOCH {epoch:>{epoch_width}} '
    bold_epoch_str = f'{BOLD_ON}{epoch_str}{BOLD_OFF}'

    logbox_len_l = (logbox_len - len(epoch_str)) // 2
    logbox_len_r = (logbox_len - len(epoch_str)) - logbox_len_l

    return f'{EPOCH_FILL_CHAR * logbox_len_r}{bold_epoch_str}{EPOCH_FILL_CHAR * (logbox_len_l)}'


def make_sec_header(sec_name: str, logbox_len: int = 100) -> str:
    '''
    Creates the string used to indicate the section (e.g. LOSS, EVAL, TIME) during logging.
    '''
    sec_str = f'{BOLD_ON}[{sec_name.upper()}]{BOLD_OFF}'
    sec_len = len(sec_name) + 2 # Length of [section]
    logbox_len_r = logbox_len - sec_len - (len(ROW_DIV_CHAR) + 1) # Right edge length after [section]
    
    return f'{ROW_DIV_CHAR} {sec_str}{ROW_DIV_CHAR:>{logbox_len_r}}'


def make_log_entry(
    name: str, 
    value: float,
    unit: Optional[str] = None,
    num_decimals: int = 4, 
    entry_len: int = 20
) -> str:
    unit = '' if unit is None else f' {unit}'
        
    # Available length for the numerical value
    # Subtract 2 b/c extra length from symbols and space
    value_len = entry_len - len(name) - len(unit) - 2
    return f'{name}: {value:>{value_len}.{num_decimals}f}{unit}'


def make_log_sec(
    sec_name: str,
    entry_names: Sequence[str],
    entry_values: Sequence[float],
    entry_units: EntryLogUnits = None,
    logbox_len: int = 100,
    max_row_entries: int = 3, 
    num_decimals: int = 4
) -> str:
    # --------------------
    # Setup
    # --------------------
    num_entries = len(entry_names) # Total number of input entries (non-pad)
    if len(entry_values) != num_entries:
        raise ValueError('Length of entry_values must match length of entry_names.')

    if (entry_units is None) or (isinstance(entry_units, str)):
        entry_units = [entry_units] * num_entries
    elif len(entry_units) != num_entries:
        raise ValueError(
            'If entry_units is a sequence of units, its length must match length of entry_names.'
        )

    # Total number of entries (including padding) to have a complete grid
    num_grid_entries = (math.ceil(num_entries / max_row_entries)) * max_row_entries

    # List to store all section strings/rows
    sec_rows = [make_sec_header(sec_name, logbox_len)] # Initialized with section header

    # Computing length of each entry
    seps_len = len(ROW_DIV_CHAR) * (max_row_entries + 1) + 2 * max_row_entries # Length from dividers and spacing
    entries_len = logbox_len - seps_len # Total length available for entries
    base_entry_len = entries_len // max_row_entries
    last_entry_len = entries_len - (max_row_entries - 1) * base_entry_len

    # --------------------
    # Making entry rows
    # --------------------
    for i in range(num_grid_entries):
        if i % max_row_entries == 0:
            entry_row = f'{ROW_DIV_CHAR} ' # Initialize opening divider for current row
        else:
            entry_row += f' {ROW_DIV_CHAR} ' # Add middle divider

        last_entry = (i + 1) % max_row_entries == 0
        entry_len = last_entry_len if last_entry else base_entry_len

        # Add entry (or padding)
        if i < num_entries:
            entry_row += make_log_entry(entry_names[i], entry_values[i], entry_units[i], 
                                        num_decimals, entry_len)
        else:
            entry_row += ' ' * entry_len # Padding for empty columns

        if last_entry:
             # Add closing divider and append to list of section strings
            sec_rows.append(f'{entry_row} {ROW_DIV_CHAR}')
            
    return '\n'.join(sec_rows)


def make_metric_log_sec(
    metric_results: HistoryResults,
    fields: MetricLogFields,
    units: EntryLogUnits = None,
    logbox_len: int = 100, 
    max_row_entries: int = 3, 
    num_decimals: int = 4
) -> str:
    num_fields = len(fields)
    if (units is None) or (isinstance(units, str)):
        units = [units] * num_fields

    # Construct a valid (non-None) list of metric names, values, and units
    metric_names, metric_values, metric_units = [], [], []
    for field, unit in zip(fields, units):
        if isinstance(field, tuple):
            key_path, agg = field
        else:
            key_path, agg = field, 'mean'

        # Traverse down key path to get metric value
        value = nested_extract(metric_results, key_path, strict = False, default = None)

        # Aggregate metric value if necessary
        if isinstance(value, torch.Tensor):
            value = apply_agg(value, agg)
            name = f'{key_path} ({agg})'
        elif isinstance(value, float):
            name = key_path
        else:
            value = None # Key_path led to an improper value

        if value is not None:
            metric_names.append(name)
            metric_values.append(value)
            metric_units.append(unit)
        else:
            warnings.warn(
                f"Skipping metric field '{field}': contains an improper key path '{key_path}' "
                 'for extracting and/or aggregating into a scalar metric (float). ' 
                 'This entry will be omitted from the validation metric logs.',
                UserWarning
            )

    # Construct evaluation section string
    return make_log_sec(
        sec_name = 'val metrics',
        entry_names = metric_names,
        entry_values = metric_values,
        entry_units = metric_units,
        logbox_len = logbox_len,
        max_row_entries = max_row_entries,
        num_decimals = num_decimals
    )