import os
import sys

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from typing import Dict, Any
import hashlib
import json

import torch
from loguru import logger
import collections.abc
import torch.distributed as dist

from omegaconf import OmegaConf, DictConfig
from pydantic.v1.utils import deep_update

from src.log_ML.json_log import to_serializable
from src.log_ML.mlflow_log import init_mlflow_logging
from src.utils.general_utils import print_dict_to_logger

BASE_CONFIG_DIR = os.path.join(os.getcwd(), '..', 'configs')
if not os.path.exists(BASE_CONFIG_DIR):
    raise IOError('Cannot find the directory for (base) .yaml config files from "{}"'.format(BASE_CONFIG_DIR))

CONFIG_DIR = os.path.join(BASE_CONFIG_DIR, 'train_configs')
if not os.path.exists(CONFIG_DIR):
    raise IOError('Cannot find the directory for (task) .yaml config files from "{}"'.format(CONFIG_DIR))


def import_config(args: dict,
                  task_config_file: str,
                  base_config_file: str = 'base_config.yaml',
                  hyperparam_name: str = None,
                  log_level: str = "INFO"):

    base_config = import_config_from_yaml(config_file=base_config_file,
                                          config_dir=BASE_CONFIG_DIR,
                                          config_type='base')

    config = update_base_with_task_config(task_config_file=task_config_file,
                                          config_dir=CONFIG_DIR,
                                          base_config=base_config)

    if args['run_mode'] != 'train':
        config = update_config_for_non_train_mode(config,
                                                  config_dir=os.path.join(BASE_CONFIG_DIR, 'runmode_configs'),
                                                  run_mode=args['run_mode'])

    config = config_manual_fixes(config)

    # Add the input arguments as an extra subdict to the config
    # config['config']['ARGS'] = args
    config['ARGS'] = args

    # Setup the computing environment
    config['config']['MACHINE'] = set_up_environment(machine_config=config['config']['MACHINE'],
                                                     local_rank=config['ARGS']['local_rank'])

    config_hash = dict_hash(dictionary=config['config'])
    start_time = get_datetime_string()

    if not config['ARGS']['s3_mount']:
        logger.warning('Skipping realtime AWS S3 write, and writing run artifacts to a local non-mounted dir')
        config['ARGS']['output_dir'] += '_local'

    logger.info('Username = {}, UID = {}, GID = {}'.format(os.getenv('USER'), os.getuid(), os.getgid()))
    debug_mounts(mounts_list=get_mounts_from_args(args=config['ARGS']))

    output_experiments_base_dir = os.path.join(config['ARGS']['output_dir'], 'experiments')
    logger.info('Save the run-specific parameters to config["run"]')
    config['run'] = {
        'hyperparam_name': hyperparam_name,
        'hyperparam_base_name': hyperparam_name,
        'output_base_dir': config['ARGS']['output_dir'],
        'output_experiments_base_dir': output_experiments_base_dir,
        'output_experiment_dir': os.path.join(output_experiments_base_dir, hyperparam_name),
        # these sit outside the experiments and are not "hyperparameter run"-specific
        'output_wandb_dir': os.path.join(config['ARGS']['output_dir'], 'WANDB'),
        'output_mlflow_dir': os.path.join(config['ARGS']['output_dir'], 'MLflow'),
        'config_hash': config_hash,
        'start_time': start_time,
        'src_dir': os.getcwd(),
        'repo_dir': os.path.join(os.getcwd(), '..'),
    }

    print_dict_to_logger(dict_in=config['run'], prefix=' ')

    # Init variables for 'run'
    config['run']['repeat_artifacts'] = {}
    config['run']['ensemble_artifacts'] = {}
    config['run']['fold_dir'] = {}

    # Get a predefined smaller subset to be logged as MLflow/WANDB columns/hyperparameters
    # to make the dashboards cleaner, or alternatively you can just dump the whole config['config']
    config['hyperparameters'] = define_hyperparam_run_params(config)
    # TODO! Fix this with the updated nesting, with some nicer recursive function for undefined depth
    config['hyperparameters_flat'] = None  # flatten_nested_dictionary(dict_in=config['hyperparameters'])

    logger.info('Save the derived hyperparameters to config["hyperparameters"]')
    logger.info(config['hyperparameters'])  # TODO add to print_the_dict_to_logger() nested dicts as well

    if config['config']['LOGGING']['unique_hyperparam_name_with_hash']:
        # i.e. whether you want a tiny change in dictionary content make this training to be grouped with
        # another existing (this is FALSE), or to be a new hyperparam name (TRUE) if you forgot for example to change
        # the hyperparam run name after some config changes. In some cases you would like to run the same experiment
        # over and over again and to be grouped under the same run if you want to know how reproducible your run is
        # and don't want to add for example date to the hyperparam name
        # e.g. from "'hyperparam_example_name'" ->
        #     "'hyperparam_example_name_9e0c146a68ec606442a6ec91265b11c3'"
        config['run']['hyperparam_name'] += '_{}'.format(config_hash)
        config['run']['output_experiment_dir'] += '_{}'.format(config_hash)
        logger.info('Unique hyperparam name with hash')
        logger.info('  hyperparam_name = "{}"'.format(config['run']['hyperparam_name']))
        logger.info('  output_experiment_dir = "{}"'.format(config['run']['output_experiment_dir']))

    log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | "
                  "<yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
    log_file = 'log_{}.txt'.format(hyperparam_name)
    config['run']['output_log_path'] = os.path.join(config['run']['output_experiment_dir'], log_file)
    try:
        logger.add(config['run']['output_log_path'],
                   level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)
    except Exception as e:
        logger.error('Problem initializing the log file to the artifacts output, permission issues?? e = {}'.format(e))
        raise IOError('Problem initializing the log file to the artifacts output, have you created one? '
                      'do you have the permissions correct? See README.md for the "minivess_mlops_artifacts" creation'
                      'with symlink to /mnt \n error msg = {}'.format(e))
    logger.info('Log (loguru) will be saved to disk to "{}"'.format(config['run']['output_log_path']))

    log_file = 'stdout_{}.txt'.format(hyperparam_name)
    config['run']['stdout_log_path'] = os.path.join(config['run']['output_experiment_dir'], log_file)
    logger.info('Stdout will be saved to disk to "{}"'.format(config['run']['stdout_log_path']))
    sys.stdout = open(config['run']['stdout_log_path'], 'w')
    sys.stderr = sys.stdout

    # Initialize ML logging (experiment tracking)
    config['run']['mlflow'], mlflow_dict = init_mlflow_logging(config=config,
                                                               mlflow_config=config['config']['LOGGING']['MLFLOW'],
                                                               experiment_name=config['ARGS']['project_name'],
                                                               run_name=config['run']['hyperparam_name'])

    return config


def config_manual_fixes(config: dict):
    """
    Quick'n'dirty fixes for YAML import
    """
    return config


def update_config_for_non_train_mode(config: dict,
                                     config_dir: str,
                                     run_mode: str):

    logger.warning('Your run_mode != "train" (it is "{}", '
                   'you are running this for dev or CI/CD purposes'.format(run_mode))

    if not os.path.exists(config_dir):
        raise IOError('Cannot find the directory for run_mode .yaml config files from "{}"'.format(config_dir))

    config_fname = '{}_config.yaml'.format(run_mode)
    config = update_base_with_task_config(task_config_file=config_fname,
                                          config_dir=config_dir,
                                          base_config=deepcopy(config))

    # TODO! add some prefix for project and/or run name so you can log the runs to MLflow/WANDB if desired
    #  without them messing up the actual training results dashboards

    return config


def update_base_with_task_config(task_config_file: str,
                                 config_dir: str,
                                 base_config: DictConfig) -> dict:

    # https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    def update_config_dictionary(d: DictConfig,
                                 u: DictConfig,
                                 input_is_omegaconf: bool = True,
                                 method: str = 'pydantic') -> DictConfig:
        if input_is_omegaconf:
            # TODO! examine OmegaConf.merge() and OmegaConf.update() for omegaConf native merge?
            d = OmegaConf.to_container(d, resolve=True)
            u = OmegaConf.to_container(u, resolve=True)

        if method == 'pydantic':
            d = deep_update(d, u)
        else:
            for k, v in u.items():
                if isinstance(v, collections.abc.Mapping):
                    d[k], _ = update_config_dictionary(d.get(k, {}), v, input_is_omegaconf=False)
                else:
                    d[k] = v

        if input_is_omegaconf:
            d = OmegaConf.create(d)

        return d

    def get_changed_keys(base_config, config):
        set1 = set(base_config.items())
        set2 = set(config.items())
        diff = set1 ^ set2
        return diff

    # Task config now contains only subset of keys (of the base config), i.e. the parameters
    # that you change (no need to redefine all possible parameters)
    task_config = import_config_from_yaml(config_file=task_config_file,
                                          config_dir=config_dir,
                                          config_type='task')

    # update the base config now with the task config (i.e. the keys that have changed)
    config = update_config_dictionary(d=base_config,
                                      u=task_config)
    # logger.info('Updated the base config with a total of {} changed keys from the task config', no_of_updates)
    # diff = diff_OmegaDicts(a=base_config, b=config) # TODO!
    # TODO! Need to check also whether you have a typo in dict, or some extra nesting, and the desired
    #  output does not happen, check VISSL for "guidance"

    # TOADD: Hydra?
    # https://www.sscardapane.it/tutorials/hydra-tutorial/#first-steps-manipulating-a-yaml-file
    # https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710
    # https://github.com/khuyentran1401/Machine-learning-pipeline

    return config


def import_config_from_yaml(config_file: str = 'base_config.yaml',
                            config_dir: str = CONFIG_DIR,
                            config_type: str = 'base',
                            load_method: str = 'OmegaConf') -> DictConfig:

    config_path = os.path.join(config_dir, config_file)
    if os.path.exists(config_path):
        logger.info('Import {} config from "{}", method = "{}"', config_type, config_path, load_method)
        if load_method == 'OmegaConf':
            # https://www.sscardapane.it/tutorials/hydra-tutorial/#first-steps-manipulating-a-yaml-file
            # https://omegaconf.readthedocs.io/en/2.3_branch/
            config = OmegaConf.load(config_path)
        else:
            raise IOError('Unknown method for handling configs? load_method = {}'.format(load_method))
    else:
        raise IOError('Cannot find {} config from = {}'.format(config_type, config_path))

    return config


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
        try:
            torch.cuda.set_device(device)
        except Exception as e:
            # e.g. "CUDA unknown error - this may be due to an incorrectly set up environment"
            raise EnvironmentError('Problem setting up the CUDA device'.format(e))
        torch.backends.cudnn.benchmark = True
    else:
        device = 'cpu'
        logger.warning('No Nvidia CUDA GPU found, training on CPU instead!')

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
    try:
        encoded = json.dumps(dictionary, sort_keys=True, default=to_serializable).encode()
        dhash.update(encoded)
        hash_out = dhash.hexdigest()
    except Exception as e:
        logger.warning('Problem getting the hash of the config dictionary, error = {}'.format(e))
        hash_out = None
    return hash_out


def get_datetime_string():

    # Use GMT time if you have coworkers across the world running the jobs
    now = datetime.now(timezone.utc)
    date = now.strftime("%Y%m%d-%H%MGMT")

    return date


def define_hyperparam_run_params(config: dict) -> dict:
    """
    To be optimized later? You could read these from the config as well, which subdicts count as
    experiment hyperparameters. Not just dumping all possibly settings that maybe not have much impact
    on the "ML science" itself, e.g. number of workers for dataloaders or something
    :param config:
    :return:
    """

    hyperparams = {}
    cfg = config['config']

    logger.info('Hand-picking the keys/subdicts from "config" that are logged as hyperparameters for MLflow/WANDB')

    # What datasets were you used for training the model
    hyperparams['datasets'] = cfg['DATA']['DATA_SOURCE']['DATASET_NAMES']

    # What model and architecture hyperparams you used
    hyperparams['models'] = {}
    model_names = cfg['MODEL']['META_MODEL']
    for model_name in model_names:
        hyperparams['models'][model_name] = {}
        hyperparams['models'][model_name]['architecture'] = cfg['MODEL'][model_name]

    hyperparams = parse_training_params(model_names, hyperparams, cfg)

    return hyperparams


def parse_training_params(model_names, hyperparams, cfg):

    # Training params
    training_tmp = deepcopy(cfg['TRAINING'])
    setting_names = ['LOSS', 'OPTIMIZER', 'SCHEDULER']
    for model_name in model_names:
        hyperparams['models'][model_name]['training'] = training_tmp[model_name]
        for setting_name in setting_names:
            settings, param_name = parse_settings_by_name(cfg=cfg, setting_name=setting_name, settings_key=setting_name)
            hyperparams['models'][model_name]['training'][setting_name] = {}
            hyperparams['models'][model_name]['training'][setting_name][param_name] = settings

    return hyperparams


def parse_settings_by_name(cfg: dict, setting_name: str, settings_key: str):
    settings_tmp = cfg[settings_key]
    # remove one nesting level
    param_name = list(settings_tmp.keys())[0]
    settings = settings_tmp[param_name]
    return settings, param_name


def flatten_nested_dictionary(dict_in: dict, delim: str = '__') -> dict:

    def parse_non_dict(var_in):
        # placeholder if you for example have lists that you would like to convert to strings?
        return var_in

    dict_out = {}
    for key1 in dict_in:
        subentry = dict_in[key1]
        if isinstance(subentry, dict):
            for key2 in subentry:
                subentry2 = subentry[key2]
                key_out2 = '{}{}{}'.format(key1, delim, key2)
                if isinstance(subentry2, dict):
                    for key3 in subentry2:
                        subentry3 = subentry2[key3]
                        key_out3 = '{}{}{}{}{}'.format(key1, delim, key2, delim, key3)
                        dict_out[key_out3] = parse_non_dict(subentry3)
                else:
                    dict_out[key_out2] = parse_non_dict(subentry2)
        else:
            dict_out[key1] = parse_non_dict(subentry)

    return dict_out


def get_mounts_from_args(args: dict) -> list:

    logger.debug('Getting mount names from the args')
    mounts = []
    mounts.append(args['data_dir'])
    mounts.append(args['output_dir'])

    return mounts


def debug_mounts(mounts_list: list,
                 try_to_write: bool = True):

    # mounts_list = ['/home/petteri/artifacts']
    for mount in mounts_list:
        logger.debug('MOUNT: {}'.format(mount))
        if not os.path.exists(mount):
            logger.error('Mount "{}" does not exist!'.format(mount))
            raise IOError('Mount "{}" does not exist!'.format(mount))
        path = Path(mount)
        owner = path.owner()
        group = path.group()
        logger.debug(f" owned by {owner}:{group} (owner:group)")
        mount_obj = os.stat(mount)
        oct_perm = oct(mount_obj.st_mode)[-4:]
        logger.debug(f" owned by {mount_obj.st_uid}:{mount_obj.st_gid} (owner:group)")
        logger.debug(f" mount permissions: {oct_perm}")
        logger.debug(' read access = {}'.format(os.access(mount, os.R_OK)))  # Check for read access
        logger.debug(' write access = {}'.format(os.access(mount, os.W_OK)))  # Check for write access
        logger.debug(' execution access = {}'.format(os.access(mount, os.X_OK)))  # Check for execution access
        logger.debug(' existence of dir = {}'.format(os.access(mount, os.F_OK)))  # Check for existence of file

        if os.access(mount, os.W_OK):
            logger.debug('Trying to write to the mount (write access was OK)')
            path_out = os.path.join(mount, 'test_write.txt')

            if os.path.exists(path_out):
                # unlike normal filesystem, mountpoint-s3 does not allow overwriting files,
                # so if it exists already we need to delete it first
                logger.info('File {} already exists, deleting it first'.format(path_out))
                try:
                    os.remove(path_out)
                except Exception as e:
                    logger.error('Problem deleting file {}, e = {}'.format(path_out, e))
                    raise IOError('Problem deleting file {}, e = {}'.format(path_out, e))

            try:
                file1 = open(path_out, "w")
                file1.write('Hello debug world!')
                file1.close()
                logger.debug('File write succesful!')
                mount_obj = os.stat(path_out)
                logger.debug(' file_permission = {}, {}:{}'.
                             format(oct(os.stat(path_out).st_mode)[-4:],
                                    mount_obj.st_uid, mount_obj.st_gid))
            except Exception as e:
                logger.error('Problem with file write to {}, e = {}'.format(path_out, e))
                raise IOError('Problem with file write to {}, e = {}'.format(path_out, e))

            if os.path.exists(path_out):
                try:
                    os.remove(path_out)
                    logger.debug('File delete succesful!')
                except Exception as e:
                    logger.error('Problem deleting file {}, e = {}'.format(path_out, e))
                    raise IOError('Problem deleting file {}, e = {}'.format(path_out, e))
            else:
                logger.debug('Weirdly you do not have any file to delete even though write went through OK?')
