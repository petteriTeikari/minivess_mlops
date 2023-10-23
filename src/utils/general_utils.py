import psutil
from loguru import logger
from omegaconf import OmegaConf


def check_if_key_in_dict(dict_in: dict, key_in: str):
    if key_in in dict_in.keys():
        return True
    else:
        return False


def diff_omegadicts(a, b, missing=KeyError):
    """
    https://stackoverflow.com/a/59141787
    Find keys and values which differ from `a` to `b` as a dict.

    If a value differs from `a` to `b` then the value in the returned dict will
    be: `(a_value, b_value)`. If either is missing then the token from
    `missing` will be used instead.

    :param a: The from dict
    :param b: The to dict
    :param missing: A token used to indicate the dict did not include this key
    :return: A dict of keys to tuples with the matching value from a and b
    """
    a = OmegaConf.to_container(a, resolve=True)
    b = OmegaConf.to_container(b, resolve=True)

    return {
        key: (a.get(key, missing), b.get(key, missing))
        for key in dict(
            set(a.items()) ^ set(b.items())
        ).keys()
    }


def print_dict_to_logger(dict_in: dict, prefix: str = ''):
    for k, v in dict_in.items():
        logger.info('{}  {}: {}'.format(prefix, k, v))
        # if isinstance(v, dict):
        #     print_dict_to_logger(v, prefix='  ')
        # else:
        #     logger.info('{}  {}: {}'.format(prefix, k, v))


def print_memory_stats_to_logger():
    svmem = psutil.virtual_memory()
    logger.debug('Memory usage: {}% ({:.2f}/{:.2f} GB)',
                 svmem.percent, svmem.used/10**9, svmem.total/10**9)
