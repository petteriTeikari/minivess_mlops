import subprocess
import time
import warnings
from platform import python_version, release, system, processor
import re
from loguru import logger

from monai.config import get_config_values, get_optional_config_values, get_system_info, get_gpu_info
import numpy as np
import psutil
import torch


def get_run_metadata():

    metadata = get_library_versions()
    sysinfo = get_system_information()
    metadata = {**metadata, **sysinfo}
    ids = get_commit_id()
    metadata = {**metadata, **ids}

    return metadata


def get_commit_id() -> dict:

    def get_git_revision_hash() -> str:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

    def get_git_revision_short_hash() -> str:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()

    # Get the current git commit id
    try:
        git_hash_short = get_git_revision_short_hash()
        git_hash = get_git_revision_hash()
    except Exception as e:
        warnings.warn('Failed to get the git hash, e = {}'.format(e))
        git_hash_short, git_hash = np.nan, np.nan

    return {'git_hash_short': git_hash_short, 'git_hash': git_hash}


def get_library_versions() -> dict:

    metadata = {}
    try:
        metadata['v_Python'] = python_version()
        metadata['v_Numpy'] = np.__version__
        metadata['v_OS'] = system()
        metadata['v_OS_kernel'] = release()  # in Linux systems
        metadata['v_Torch'] = str(torch.__version__)
        # https://www.thepythoncode.com/article/get-hardware-system-information-python
    except Exception as e:
        warnings.warn('Problem getting library versions, error = {}'.format(e))

    try:
        metadata['monai'] = get_monai_config()
    except Exception as e:
        warnings.warn('Problem with the MONAI printouts, error = {}'.format(e))

    try:
        metadata['v_CUDA'] = torch.version.cuda
        metadata['v_CuDNN'] = torch.backends.cudnn.version()
    except Exception as e:
        warnings.warn('Problem getting CUDA library versions, error = {}'.format(e))

    return metadata


def get_monai_config():

    monai_dict = {}

    monai_dict['MONAI_libs'] = get_optional_config_values()
    monai_dict['libs'] = get_config_values()
    monai_dict['system'] = get_system_info()
    monai_dict['GPU'] = get_gpu_info()

    logger.info('MONAI | LIBRARY VERSIONS:')
    for k, v in monai_dict['libs'].items():
        logger.info('  {}: {}'.format(k, v))

    logger.info('MONAI | OPTIONAL LIBRARY VERSIONS:')
    for k, v in monai_dict['MONAI_libs'].items():
        logger.info('  {}: {}'.format(k, v))

    logger.info('MONAI | SYSTEM:')
    for k, v in monai_dict['system'].items():
        logger.info('  {}: {}'.format(k, v))

    logger.info('MONAI | GPU:')
    for k, v in monai_dict['GPU'].items():
        logger.info('  {}: {}'.format(k, v))

    return monai_dict


def get_system_information() -> dict:

    metadata = {}
    try:
        # https://stackoverflow.com/questions/4842448/getting-processor-information-in-python
        metadata['sys_cpu'] = get_processor_info()
        metadata['sys_RAM_GB'] = str(round(psutil.virtual_memory().total / (1024 ** 3), 1))
    except Exception as e:
        warnings.warn('Problem getting system info, error = {}'.format(e))

    return metadata


def get_processor_info():

    model_name = np.nan

    if system() == "Windows":
        all_info = processor()
        # cpuinfo better? https://stackoverflow.com/a/62888665
        warnings.warn('You need to add to Windows parsing for your CPU name')

    elif system() == "Darwin":
        all_info = subprocess.check_output(['/usr/sbin/sysctl', "-n", "machdep.cpu.brand_string"]).strip()
        warnings.warn('You need to add to Mac parsing for your CPU name')

    elif system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                model_name = re.sub(".*model name.*:", "", line, 1)

    else:
        warnings.warn('Unknown OS = {}, cannot get the CPU name'.format(system()))

    return model_name