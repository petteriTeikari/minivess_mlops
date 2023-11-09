import os

import psutil
from loguru import logger
from omegaconf import OmegaConf
from pathlib import Path


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
        for key in dict(set(a.items()) ^ set(b.items())).keys()
    }


def print_dict_to_logger(dict_in: dict, prefix: str = ""):
    for k, v in dict_in.items():
        logger.info("{}{}: {}".format(prefix, k, v))
        # if isinstance(v, dict):
        #     print_dict_to_logger(v, prefix='  ')
        # else:
        #     logger.info('{}  {}: {}'.format(prefix, k, v))


def print_memory_stats_to_logger():
    svmem = psutil.virtual_memory()
    logger.debug(
        "Memory usage: {}% ({:.2f}/{:.2f} GB)",
        svmem.percent,
        svmem.used / 10**9,
        svmem.total / 10**9,
    )


def is_docker():
    cgroup = Path("/proc/self/cgroup")
    return (
        Path("/.dockerenv").is_file()
        or cgroup.is_file()
        and "docker" in cgroup.read_text()
    )


def import_from_dotenv(repo_dir: str):
    # Manual import instead of the python-dotenv package which seemed possibly finicky to get installed
    env_path = os.path.join(repo_dir, ".env")
    if not os.path.exists(env_path):
        # This does not come with the cloned repo as these are sensitive data and
        # specific to your environment
        logger.warning("No .env file found at {}, skipping import".format(env_path))
    else:
        logger.info("Importing .env file from {}".format(env_path))
        with open(env_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.strip().split("=", 1)
                os.environ[key] = value


def get_path_size(start_path: str = "."):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    total_size_kB = total_size / 1024

    return total_size_kB
