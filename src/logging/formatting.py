#####################################
# Imports & Dependencies
#####################################
import math

from typing import Optional, Sequence, TypeAlias, Union, Dict

from src.metrics.types import MetricResults
from src.metrics.postprocess import MetricSpec, select_and_agg_scalar_metric


BOLD_ON = '\033[1m'
BOLD_OFF = '\033[0m'
EPOCH_FILL_CHAR = '='
SEC_DIV_CHAR = '-'
ROW_DIV_CHAR = '|'

EntryLogUnits: TypeAlias = Union[str, Sequence[Optional[str]]]


#####################################
# Functions
#####################################
def make_epoch_header(epoch: int, logbox_len: int = 100, epoch_width: int = 3) -> str:
    '''
    Creates a formatted epoch header to indicate the epoch index during logging.

    Args:
        epoch (int): 
            Epoch index in the header.
        logbox_len (int): 
            Total character width of the epoch header.
            Default is `100`.
        epoch_width (int): 
            Minimum character width occupied by the epoch index in the header.
            Default is `3`.

    Returns:
        str: 
            The formatted epoch header.

    Example:
        ```
        ============================================= EPOCH  0 ============================================
        ```
    '''    
    epoch_str = f' EPOCH {epoch:>{epoch_width}} '
    bold_epoch_str = f'{BOLD_ON}{epoch_str}{BOLD_OFF}'

    logbox_len_l = (logbox_len - len(epoch_str)) // 2
    logbox_len_r = (logbox_len - len(epoch_str)) - logbox_len_l

    return f'{EPOCH_FILL_CHAR * logbox_len_r}{bold_epoch_str}{EPOCH_FILL_CHAR * (logbox_len_l)}'


def make_sec_header(sec_name: str, logbox_len: int = 100) -> str:
    '''
    Creates a formatted section header to indicate the section 
    (e.g. LOSS, EVAL, TIME) during logging.

    Args:
        sec_name (str): 
            The section name. This is displayed in all uppercases.
        logbox_len (int): 
            Total character width of the section header.
            Default is `100`.

    Returns:
        str: 
            The formatted section header.

    Example:
        ```
        | [SEC_NAME]                                                                                       |
        ```
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
    '''
    Creates a formatted log entry to display numerical logging information 
    (e.g. loss, metric, or time values).

    Args:
        name (str): 
            Name of the value being logged.
        value (float): 
            Value associated with `name`.
        unit (optional, str): 
            Unit assocated with `value`.
            If `None`, the value is assumed to be unitless.
        num_decimals (int): 
            Maximum number of decimals to display for `value`.
            Default is `4`.
        entry_len (int):
            Minimum character length of the log entry.
            Default is `20`.

    Returns:
        str: 
            The formatted log entry.

    Example:
        ```
        | Name:           Value Unit |
        ```
    '''
    unit = '' if unit is None else f' {unit}'
        
    # Available length for the numerical value
    # Subtract 2 b/c extra length from symbols and space
    value_len = entry_len - len(name) - len(unit) - 2
    return f'{name}: {value:>{value_len}.{num_decimals}f}{unit}'


def make_log_sec(
    sec_name: str,
    entry_names: Sequence[str],
    entry_values: Sequence[float],
    entry_units: Optional[EntryLogUnits] = None,
    logbox_len: int = 100,
    max_row_entries: int = 3, 
    num_decimals: int = 4
) -> str:
    '''
    Creates a formatted log section containing multiple log entries
    to display a grouped numerical logging information
    (e.g. loss, metric, or time values).

    Note: All sequence inputs must be the same length.

    Args:
        sec_name (str): 
            The section name. This is displayed in all uppercases.
        entry_names (Sequence[str]): 
            Sequence of names for values being displayed in the logging section.
        entry_values (Sequence[str]): 
            Sequence of values associated with `entry_names`.
        entry_units (optional, EntryLogUnits): 
            Sequence of units associated with `entry_values`.
            If an element is `None`, the associated value is assumed to be unitless.
            If `None`, all values are assumed to be unitless.
        logbox_len (int): 
            Total character width of the section.
            Default is `100`.
        max_row_entries (int): 
            Maximum number of entries in a row of the section.
            The length of each section entry will be roughly `logbox_len//max_row_entries`.
            Default is `3`.
        num_decimals (int): 
            Maximum number of decimals to display for each entry value.
            Default is `4`.
    
    Returns:
        str: 
            The formatted log section.

    Example:
        ```
        | [SEC_NAME]                                                                                    |
        | Name_0:        Value_0 Unit_0 | Name_1:        Value_1 Unit_1 | Name_2:        Value_2 Unit_2 |
        | Name_3:        Value_3 Unit_3 | Name_4:        Value_4 Unit_4 | Name_5:        Value_5 Unit_5 |
        | Name_6:        Value_6 Unit_6 | Name_7:        Value_7 Unit_7 | Name_8:        Value_8 Unit_8 |
        ```
    '''
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
    metric_results: MetricResults,
    metric_specs: Dict[str, MetricSpec],
    logbox_len: int = 100, 
    max_row_entries: int = 3, 
    num_decimals: int = 4
) -> str:
    '''
    Creates a formatted log section for metric information.

    Note: See `src.metrics.types` for an example structure of `MetricResults`.
    Note: Each metric specification (`MetricSpec`) must produce a
          scalar value after extracting and optionally subsetting and/or aggregating
          the metric from the provided `MetricResults`.

    Args:
        metric_results (MetricResults): 
            Metric result dictionary containing the metrics to display.
        metric_specs (Dict[str, MetricSpec]): 
            Dictionary mapping display names to metric specifications (`MetricSpec`)
            indicating what metrics to display.
        logbox_len (int):
            Total character width of the section.
            Default is `100`.
        max_row_entries (int): 
            Maximum number of entries in a row of the section.
            The length of each section entry will be roughly `logbox_len//max_row_entries`.
            Default is `3`.
        num_decimals (int): 
            Maximum number of decimals to display for each metric value.
            Default is `4`.
    
    Returns:
        str: 
            The formatted log section for metric information.

    Example:
        ```
        | [VAL METRICS]                                                                                    |
        | Accuracy:               0.7526 | Recall (Mean):          0.7526 | Precision (Mean):       0.7560 |
        | F1-score (Mean):        0.7514 | Recall (Min):           0.4200 | Precision (Min):        0.4486 |
        | F1-score (Min):         0.4638 |                                |                                |
        ```
    '''
    # Construct list of metric names, values, and units
    metric_names = list(metric_specs.keys())
    metric_values, metric_units = [], []
    for spec in metric_specs.values():
        # Get specified metric
        metric_values.append(
            select_and_agg_scalar_metric(
                metric_results, 
                key_path = spec.key_path,
                class_idxs = spec.class_idxs,
                agg = spec.agg
            )
        )
        metric_units.append(spec.unit)

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