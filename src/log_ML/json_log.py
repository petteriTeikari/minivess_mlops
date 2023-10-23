# To dump results dictionaries to disk as JSON so that numpy arrays are saved correctly
# https://ellisvalentiner.com/post/serializing-numpyfloat32-json/
# (some more background from https://stackoverflow.com/a/64155446)
import datetime
import numpy as np
from functools import singledispatch


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


@to_serializable.register(datetime.datetime)
def ts_datetime(val):
    """Used if *val* is an instance of datetime.datetime."""
    return val.isoformat() + "Z"
