import os
import warnings

from typing import Dict, Any
import hashlib
import json

import torch
import yaml
from loguru import logger
import collections.abc
import torch.distributed as dist

from src.log_ML.json_log import to_serializable

CONFIG_DIR = os.path.join(os.getcwd(), 'configs')
if not os.path.exists(CONFIG_DIR):
    raise IOError('Cannot find the directory for (task) .yaml config files from "{}"'.format(CONFIG_DIR))
BASE_CONFIG_DIR = os.path.join(CONFIG_DIR, 'base')
if not os.path.exists(BASE_CONFIG_DIR):
    raise IOError('Cannot find the directory for (base) .yaml config files from "{}"'.format(BASE_CONFIG_DIR))


def import_config(args, task_config_file: str, base_config_file: str = 'base_config.yaml',
                  hyperparam_name: str = None, log_level: str = "INFO"):

    # TO-OPTIMIZE: Switch to Hydra here?
    base_config = import_config_from_yaml(config_file = base_config_file,
                                          config_dir = BASE_CONFIG_DIR,
                                          config_type = 'base')

    config = update_base_with_task_config(task_config_file = task_config_file,
                                          config_dir = CONFIG_DIR,
                                          base_config = base_config)

    # Add the input arguments as an extra subdict to the config
    # config['config']['ARGS'] = args
    config['ARGS'] = args

    # Setup the computing environment
    config['config']['MACHINE'] = set_up_environment(machine_config=config['config']['MACHINE'],
                                                     local_rank=config['ARGS']['local_rank'])

    config['run'] = {
        'hyperparam_name': hyperparam_name,
        'output_artifacts_dir': os.path.join(config['ARGS']['output_dir'], hyperparam_name),
        # 'config_hash': dict_hash(config['config'])
    }

    log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | "
                  "<yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
    logger.add(os.path.join(config['run']['output_artifacts_dir'], 'log_{}.txt'.format(hyperparam_name)),
               level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)
    logger.info('Log will be saved to disk to "{}"'.format(config['run']['output_artifacts_dir']))

    return config


def update_base_with_task_config(task_config_file: str, config_dir: str, base_config: dict):

    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    def update_config_dictionary(d, u):
        no_of_updates = 0
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k], _ = update_config_dictionary(d.get(k, {}), v)
                no_of_updates += 1
            else:
                d[k] = v
        return d, no_of_updates

    # Task config now contains only subset of keys (of the base config), i.e. the parameters
    # that you change (no need to redefine all possible parameters)
    task_config = import_config_from_yaml(config_file = task_config_file,
                                          config_dir = CONFIG_DIR,
                                          config_type = 'task')

    # update the base config now with the task config (i.e. the keys that have changed)
    config, no_of_updates = update_config_dictionary(d = base_config, u = task_config)
    logger.info('Updated the base config with a total of {} changed keys from the task config', no_of_updates)

    return config


def import_config_from_yaml(config_file: str = 'base_config.yaml',
                            config_dir: str = CONFIG_DIR,
                            config_type: str = 'base'):

    config_path = os.path.join(config_dir, config_file)
    if os.path.exists(config_path):
        logger.info('Import {} config from "{}"', config_type, config_path)
        config = import_yaml_file(config_path)
    else:
        raise IOError('Cannot find {} config from = {}'.format(config_type, config_path))

    return config


def import_yaml_file(yaml_path: str):

    # TOADD! add scientifc notation resolver? e.g. for lr https://stackoverflow.com/a/30462009/6412152
    with open(yaml_path) as file:
        try:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    if 'cfg' not in locals():
        raise IOError('YAML import failed! See the the line and columns above that were not parsed correctly!\n'
                      '\t\tI assume that you added or modified some entries and did something illegal there?\n'
                      '\t\tMaybe a "=" instead of ":"?\n'
                      '\t\tMaybe wrong use of "’" as the closing quote?')

    return cfg


def set_up_environment(machine_config: dict, local_rank: int = 0):

    if machine_config['DISTRIBUTED']:
        # initialize the distributed training process, every GPU runs in a process
        # see e.g.
        # https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_model_training_guide.md
        # https://github.com/Project-MONAI/tutorials/blob/main/acceleration/distributed_training/brats_training_ddp.py
        dist.init_process_group(backend="nccl", init_method="env://")

    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]

    if len(available_gpus) > 0:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        warnings.warn('No Nvidia CUDA GPU found, training on CPU instead!')

    # see if this is actually the best way to do things, as "parsed" things start to be added to a static config dict
    machine_config['IN_USE'] = {'device': device,
                                'local_rank': local_rank}

    return machine_config


def hash_config_dictionary(dict_in: dict):
    """
    To check whether dictionary has changed
    https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
    :param dict_in:
    :return:
    """

def dict_hash(dictionary: Dict[str, Any]) -> str:
    """
    MD5 hash of a dictionary.
    https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
    """
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    # Fix this with the "to_serializable" TypeError: Object of type int64 is not JSON serializable
    encoded = json.dumps(dictionary, sort_keys=True, default=to_serializable).encode()
    dhash.update(encoded)
    return dhash.hexdigest()