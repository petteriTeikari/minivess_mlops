import os
import sys
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Dict, Any
import hashlib

from hydra import initialize, compose
from loguru import logger

import torch
import torch.distributed as dist
from omegaconf import DictConfig

from tests.env.mount_tests import debug_mounts
from src.log_ML.json_log import to_serializable
from src.log_ML.mlflow_log import init_mlflow_logging
from src.utils.general_utils import print_dict_to_logger
from src.utils.metadata_utils import get_run_metadata


def get_changed_keys(base_config, config):
    set1 = set(base_config.items())
    set2 = set(config.items())
    diff = set1 ^ set2
    return diff


def get_hydra_config_paths(task_cfg_name: str = None,
                           reprod_cfg_name: str = None,
                           parent_dir_string: str = '..') -> dict:

    def get_config_path():
        absolute_config_dir = os.path.join(os.getcwd(), parent_dir_string, 'configs')
        if not os.path.exists(absolute_config_dir):
            logger.error('Configuration dir "{}" does not exist!'.format(absolute_config_dir))
            raise IOError('Configuration dir "{}" does not exist!'.format(absolute_config_dir))
        relative_config_dir = os.path.join(parent_dir_string, 'configs')
        return absolute_config_dir, relative_config_dir

    def get_task_cfg_overrides_list(task_cfg_name: str, abs_cfg_path: str):
        subdir, fname = os.path.split(task_cfg_name)
        if len(subdir) == 0:
            logger.error('No subdir of the config detected in "{}" (should be like "train/cfg")'.format(task_cfg_name))
            raise IOError('No subdir of the config detected in "{}" (should be like "train/cfg")'.format(task_cfg_name))
        absolute_cfg_path = os.path.join(abs_cfg_path, subdir, f'{fname}.yaml')
        if not os.path.exists(absolute_cfg_path):
            logger.error('Task cfg "{}" does not exist!'.format(absolute_cfg_path))
            raise IOError('Task cfg "{}" does not exist!'.format(absolute_cfg_path))
        overrides_list = [f'+{subdir}={fname}']
        return overrides_list, absolute_cfg_path

    if reprod_cfg_name is not None:
        logger.info('You had provided "reprod_cfg" = "{}"! '
                    'Using the config as it is for reproduction of previous run without "overrides"'.
                    format(reprod_cfg_name))
        raise NotImplementedError('TODO! Implement this!')

    else:
        abs_cfg_path, config_path = get_config_path()
        overrides_list, abs_task_cfg_path = (
            get_task_cfg_overrides_list(task_cfg_name=task_cfg_name,
                                        abs_cfg_path=abs_cfg_path))

        return {'config_path': config_path,
                'abs_cfg_path': abs_cfg_path,
                'abs_task_cfg_path': abs_task_cfg_path,
                'overrides_list': overrides_list}


def update_task_cfg_for_ci_cd_jobs(run_mode: str = 'test_dataload',
                                   cicd_dir_name: str = 'runmode_configs') -> str:
    task_cfg_name = f'{cicd_dir_name}/{run_mode}_config'
    logger.info('Run mode = "{}", updating the task config to "{}"'.format(run_mode, task_cfg_name))
    return task_cfg_name


def hydra_import_config(hydra_cfg_paths: dict,
                        base_cfg_name: str = 'defaults',
                        job_name: str = None,
                        parent_dir_string_defaults: str = '..',
                        hydra_version_base = '1.2') -> DictConfig:
    try:
        # This relative path is from config_utils.py, so one "../" added
        with initialize(config_path=os.path.join(parent_dir_string_defaults, hydra_cfg_paths['config_path']),
                        job_name=job_name,
                        version_base=hydra_version_base):
            cfg = compose(config_name=base_cfg_name,
                          overrides=hydra_cfg_paths['overrides_list'])
            logger.info('Initializing Hydra with config_path = "{}", job_name = "{}", version_base = "{}"'.
                        format(hydra_cfg_paths['abs_cfg_path'], job_name, hydra_version_base))
            logger.info('Hydra overrides list = {}'.format(hydra_cfg_paths['overrides_list']))
    except Exception as e:
        logger.error('Problem initializing Hydra, e = {}'.format(e))
        raise IOError('Problem initializing Hydra, e = {}'.format(e))

    return cfg


def config_import_script(task_cfg_name: str = None,
                         reprod_cfg_name: str = None,
                         parent_dir_string: str = '..',
                         parent_dir_string_defaults: str = '..',
                         job_name: str = 'config_test',
                         base_cfg_name: str = 'defaults',
                         hydra_version_base = '1.2') -> DictConfig:

    # Define the paths needed to create the Hydra config
    hydra_cfg_paths = get_hydra_config_paths(task_cfg_name=task_cfg_name,
                                             reprod_cfg_name=reprod_cfg_name,
                                             parent_dir_string=parent_dir_string)

    # Compose the Hydra config by merging base and "task" config
    config = hydra_import_config(hydra_cfg_paths=hydra_cfg_paths,
                                 job_name=job_name,
                                 base_cfg_name=base_cfg_name,
                                 parent_dir_string_defaults=parent_dir_string_defaults,
                                 hydra_version_base=hydra_version_base)

    return config

def import_config(args: dict,
                  task_cfg_name: str = 'train_configs/train_task_test',
                  reprod_cfg_name: str = None,
                  log_level: str = 'DEBUG'):

    # CI/CD tasks have hard-coded "task configs"
    if args['run_mode'] != 'train':
        task_cfg_name = update_task_cfg_for_ci_cd_jobs(run_mode=args['run_mode'])

    config = config_import_script(task_cfg_name=task_cfg_name,
                                  reprod_cfg_name=reprod_cfg_name,
                                  job_name=args['project_name'])

    exp_run = set_up_experiment_run(config=config,
                                    args=args,
                                    log_level=log_level)

    return config, exp_run


def set_up_experiment_run(config: DictConfig,
                          args: dict,
                          log_level: str = 'DEBUG') -> dict:

    # Add the input arguments as an extra subdict to the config
    hyperparam_name = config['NAME']
    exp_run = {'ARGS': args}

    # Setup the computing environment
    exp_run['MACHINE'] = set_up_environment(machine_config=config['config']['MACHINE'],
                                            local_rank=args['local_rank'])

    # If you do not want to mount the artifacts directory, and find easier just to "aws sync" at the end
    if not args['s3_mount']:
        logger.warning('Skipping realtime AWS S3 write, and writing run artifacts to a local non-mounted dir')
        args['output_dir'] += '_local'
    debug_mounts(mounts_list=get_mounts_from_args(args=args))

    # Set-up run parameters
    exp_run['RUN'] = set_up_run_params(config=config,
                                       args=args,
                                       hyperparam_name=hyperparam_name)

    # Define hyperparameters for Experimnent Tracking (MLflow/WANDB)
    exp_run['HYPERPARAMETERS'], exp_run['HYPERPARAMETERS_FLAT'] = (
        define_hyperparams_from_config(config))

    # Add hash to names to make them unique
    if config['config']['LOGGING']['unique_hyperparam_name_with_hash']:
        exp_run['RUN'] = use_dict_hash_in_names(run_params=exp_run['RUN'],
                                                config_hash=exp_run['RUN']['config_hash'])

    # Set-up the log files (loguru, stdout/stderr)
    exp_run = set_up_log_files(config=config,
                               exp_run=exp_run,
                               hyperparam_name=hyperparam_name,
                               log_level=log_level)

    # Add run/environment-dependent metadata (e.g. library versions, etc.)
    exp_run['METADATA'] = get_run_metadata()

    # Initialize ML logging (experiment tracking)
    exp_run['MLFLOW'], mlflow_dict = init_mlflow_logging(config=config,
                                                         exp_run=exp_run,
                                                         mlflow_config=config['config']['LOGGING']['MLFLOW'],
                                                         experiment_name=args['project_name'],
                                                         run_name=exp_run['RUN']['hyperparam_name'])

    logger.info('Done setting up the experiment run parameters')

    return exp_run


def get_repo_dir(return_src: bool = False) -> str:
    # quick and dirty
    try:
        repo_dir = os.path.join(os.getcwd())
        base_path, last_subdir = os.path.split(repo_dir)
        if last_subdir == 'src':
            if return_src:
                repo_dir = repo_dir
            else:
                repo_dir = base_path
        else:
            if return_src:
                repo_dir = os.path.join(repo_dir, 'src')
    except Exception as e:
        raise IOError('Problem getting the repo dir (src={}), error = {}'.format(return_src, e))

    return repo_dir


def set_up_run_params(config: dict,
                      args: dict,
                      hyperparam_name: str):

    # Create hash from config dict
    config_hash = dict_hash(dictionary=config['config'])
    start_time = get_datetime_string()

    output_experiments_base_dir = os.path.join(args['output_dir'], 'experiments')
    logger.info('Save the run-specific parameters to exp_run["run"]')
    run_params = {
        'hyperparam_name': hyperparam_name,
        'hyperparam_base_name': hyperparam_name,
        'output_base_dir': args['output_dir'],
        'output_experiments_base_dir': output_experiments_base_dir,
        'output_experiment_dir': os.path.join(output_experiments_base_dir, hyperparam_name),
        # these sit outside the experiments and are not "hyperparameter run"-specific
        'output_wandb_dir': os.path.join(args['output_dir'], 'WANDB'),
        'output_mlflow_dir': os.path.join(args['output_dir'], 'MLflow'),
        'config_hash': config_hash,
        'start_time': start_time,
        'src_dir': get_repo_dir(return_src=True),
        'repo_dir': get_repo_dir()
    }
    # Init variables for 'run'
    run_params['repeat_artifacts'] = {}
    run_params['ensemble_artifacts'] = {}
    run_params['fold_dir'] = {}
    print_dict_to_logger(dict_in=run_params, prefix=' ')

    return run_params


def use_dict_hash_in_names(run_params: dict,
                           config_hash: str) -> dict:

    run_params['hyperparam_name'] += '_{}'.format(config_hash)
    run_params['output_experiment_dir'] += '_{}'.format(config_hash)
    logger.info('Unique hyperparam name with hash')
    logger.info('  hyperparam_name = "{}"'.format(run_params['hyperparam_name']))
    logger.info('  output_experiment_dir = "{}"'.format(run_params['output_experiment_dir']))

    return run_params


def define_hyperparams_from_config(config: dict) -> dict:

    # Get a predefined smaller subset to be logged as MLflow/WANDB columns/hyperparameters
    # to make the dashboards cleaner, or alternatively you can just dump the whole config['config']
    hyperparameters = define_hyperparam_run_params(config)

    # TODO! Fix this with the updated nesting, with some nicer recursive function for undefined depth
    hyperparameters_flat = {'placeholder_key': 'value'}  # flatten_nested_dictionary(dict_in=config['hyperparameters'])
    logger.info('Save the derived hyperparameters to config["hyperparameters"]')
    logger.info(hyperparameters)  # TODO add to print_the_dict_to_logger() nested dicts as well

    return hyperparameters, hyperparameters_flat


def set_up_log_files(config: DictConfig,
                     exp_run: dict,
                     hyperparam_name: str,
                     log_level: str = 'DEBUG'):

    log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | "
                  "<yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
    log_file = 'log_{}.txt'.format(hyperparam_name)
    exp_run['RUN']['output_log_path'] = os.path.join(exp_run['RUN']['output_experiment_dir'], log_file)
    try:
        logger.add(exp_run['RUN']['output_log_path'],
                   level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)
    except Exception as e:
        logger.error('Problem initializing the log file to the artifacts output, permission issues?? e = {}'.format(e))
        raise IOError('Problem initializing the log file to the artifacts output, have you created one? '
                      'do you have the permissions correct? See README.md for the "minivess_mlops_artifacts" creation'
                      'with symlink to /mnt \n error msg = {}'.format(e))
    logger.info('Log (loguru) will be saved to disk to "{}"'.format(exp_run['RUN']['output_log_path']))

    log_file = 'stdout_{}.txt'.format(hyperparam_name)
    exp_run['RUN']['stdout_log_path'] = os.path.join(exp_run['RUN']['output_experiment_dir'], log_file)
    logger.info('Stdout will be saved to disk to "{}"'.format(exp_run['RUN']['stdout_log_path']))
    sys.stdout = open(exp_run['RUN']['stdout_log_path'], 'w')
    sys.stderr = sys.stdout

    return exp_run


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
    machine_config = {'device': device,
                      'local_rank': local_rank}

    return machine_config


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
    if training_tmp['SKIP_TRAINING']:
        logger.warning('Skipping the training, not parsing training hyperparameters')
        hyperparams = {}

    else:
        setting_names = ['LOSS', 'OPTIMIZER', 'SCHEDULER']
        for model_name in model_names:
            hyperparams['models'][model_name]['training'] = training_tmp[model_name]
            hyperparams['models'][model_name]['training_extra'] = {}
            for setting_name in setting_names:
                try:
                    settings, param_name = (
                        parse_settings_by_name(model_cfg=training_tmp[model_name][setting_name]))
                    hyperparams['models'][model_name]['training_extra'][setting_name] = {}
                    hyperparams['models'][model_name]['training_extra'][setting_name][param_name] = settings
                except Exception as e:
                    logger.error('Problem parsing the training settings, model_name = {}, setting_name = {}, e = {}'.
                                 format(model_name, setting_name, e))
                    raise IOError('Problem parsing the training settings, model_name = {}, setting_name = {}, e = {}'.
                                  format(model_name, setting_name, e))

    return hyperparams


def parse_settings_by_name(model_cfg: dict):
    params = model_cfg.get('PARAMS')
    name = model_cfg.get('NAME')
    return params, name


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



