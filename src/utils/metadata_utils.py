import subprocess
import time
import warnings
from platform import python_version, release, system, processor
import re
from loguru import logger

from monai.config import print_config, print_system_info, print_gpu_info
import numpy as np
import psutil
import torch


def get_run_metadata():

    metadata = get_library_versions()
    sysinfo = get_system_information()
    metadata = {**metadata, **sysinfo}

    return metadata


def get_library_versions():

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
        logger.info('MONAI | config printouts:')
        time.sleep(0.05)
        print_config()
        print_system_info()
        print_gpu_info()
    except Exception as e:
        warnings.warn('Problem with the MONAI printouts, error = {}'.format(e))

    try:
        metadata['v_CUDA'] = torch.version.cuda
        metadata['v_CuDNN'] = torch.backends.cudnn.version()
    except Exception as e:
        warnings.warn('Problem getting CUDA library versions, error = {}'.format(e))

    return metadata


def get_system_information():

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