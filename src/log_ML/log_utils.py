import warnings

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