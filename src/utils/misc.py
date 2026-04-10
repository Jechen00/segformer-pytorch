#####################################
# Imports & Dependencies
#####################################
from typing import Union, Any

#####################################
# Functions
#####################################
def make_tuple(x: Union[Any, tuple]) -> tuple:
    '''
    Converts input to a tuple (x, x), if it is not already a tuple. 

    Args:
        x (Union[Any, tuple]): Input to convert into a tuple (if needed).
    '''
    if not isinstance(x, tuple):
        return (x, x)
    else:
        return x