import subprocess
from platform import python_version, release, system, processor
import re
from loguru import logger

from monai.config import get_config_values, get_optional_config_values, get_system_info, get_gpu_info
import numpy as np
import psutil
import torch


def get_run_metadata(metadata_method: str = 'MONAI'):

    if metadata_method == 'MONAI':
        # MONAI already comes with functions to return you library versions and system information
        metadata = get_monai_config()

    elif format == 'custom_subset':
        metadata = get_library_versions()
        sysinfo = get_system_information()
        metadata = {**metadata, **sysinfo}
    else:
        raise NotImplementedError('Unknown metadata_method to get metadata for logging = "{}"'.format(metadata_method))

    ids = get_commit_id()
    metadata = {**metadata, **ids}

    # TODO! Add DVC Library info

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
        logger.warning('Failed to get the git hash, e = {}'.format(e))
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
        logger.warning('Problem getting library versions, error = {}'.format(e))

    try:
        metadata['v_CUDA'] = torch.version.cuda
        metadata['v_CuDNN'] = torch.backends.cudnn.version()
    except Exception as e:
        logger.warning('Problem getting CUDA library versions, error = {}'.format(e))

    return metadata


def get_monai_config(flatten_dict: bool = True, clean_for_omegaconf: bool = True):

    monai_dict = {'MONAI_libs': get_optional_config_values(),
                  'libs': get_config_values(),
                  'system': get_system_info(),
                  'GPU': get_gpu_info()}

    # Manual fix for Pytorch version as it comes out as "TorchVersion" and not as a str that would work with OmegaConf
    if 'Pytorch' in monai_dict['libs']:
        monai_dict['libs']['Pytorch'] = str(monai_dict['libs']['Pytorch'])

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

    if flatten_dict:
        monai_dict = {**monai_dict['MONAI_libs'], **monai_dict['libs'], **monai_dict['system'], **monai_dict['GPU']}

    if clean_for_omegaconf:
        monai_dict = clean_monai_dict_for_omegaconf(monai_dict)

    return monai_dict


def clean_monai_dict_for_omegaconf(monai_dict):

    dict_out = {}
    for key_in in list(monai_dict.keys()):
        value = monai_dict[key_in]
        if isinstance(value, (str, int, float)):
            dict_out[key_in] = value
        else:
            logger.warning('MONAI metadata key = "{}" not cleaned to OmegaConf as its type was too "exotic", '
                           'type = {} (you see these on log though)'.format(key_in, type(value)))

    return dict_out


def get_system_information() -> dict:

    metadata = {}
    try:
        # https://stackoverflow.com/questions/4842448/getting-processor-information-in-python
        metadata['sys_cpu'] = get_processor_info()
        metadata['sys_RAM_GB'] = str(round(psutil.virtual_memory().total / (1024 ** 3), 1))
    except Exception as e:
        logger.warning('Problem getting system info, error = {}'.format(e))

    return metadata


def get_processor_info():

    model_name = np.nan

    if system() == "Windows":
        all_info = processor()
        # cpuinfo better? https://stackoverflow.com/a/62888665
        logger.warning('You need to add to Windows parsing for your CPU name')

    elif system() == "Darwin":
        all_info = subprocess.check_output(['/usr/sbin/sysctl', "-n", "machdep.cpu.brand_string"]).strip()
        logger.warning('You need to add to Mac parsing for your CPU name')

    elif system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                model_name = re.sub(".*model name.*:", "", line, 1)

    else:
        logger.warning('Unknown OS = {}, cannot get the CPU name'.format(system()))

    return model_name
