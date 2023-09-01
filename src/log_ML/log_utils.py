import json
import warnings
from functools import singledispatch

import numpy as np

def convert_value_to_numpy_array(value_in):

    if type(value_in) is np.ndarray:
        # if is "arrays", and already a Numpy array
        raise NotImplementedError('Implement when input is already np.ndarray!')

    elif isinstance(value_in, (float, int, np.floating, np.integer)):
        value_in = np.array(value_in)[np.newaxis]

    else:
        # e.g. if you save strings or something else here that is hard really to use for stats computations
        warnings.warn('Unknown value type = {}'.format(type(value_in)))

    return value_in


def compute_numpy_stats(value_array_in: np.ndarray):

    stats_dict = {}
    no_of_folds, no_of_repeats = value_array_in.shape

    stats_dict['mean'] = np.mean(value_array_in)
    stats_dict['stdev'] = np.std(value_array_in)
    stats_dict['n'] = value_array_in.size

    return stats_dict
