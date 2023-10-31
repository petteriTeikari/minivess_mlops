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
from omegaconf import DictConfig, OmegaConf

from src.utils.dict_utils import put_to_dict, cfg_key
from tests.env.mount_tests import debug_mounts
from src.log_ML.json_log import to_serializable
from src.log_ML.mlflow_log import init_mlflow_logging
from src.utils.general_utils import print_dict_to_logger, is_docker
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
                  log_level: str = 'DEBUG') -> Dict:

    # CI/CD tasks have hard-coded "task configs"
    if args['run_mode'] != 'train':
        task_cfg_name = update_task_cfg_for_ci_cd_jobs(run_mode=args['run_mode'])

    hydra_cfg = config_import_script(task_cfg_name=task_cfg_name,
                                  reprod_cfg_name=reprod_cfg_name,
                                  job_name=args['project_name'])

    run_params = set_up_experiment_run(hydra_cfg=hydra_cfg,
                                       args=args,
                                       log_level=log_level)

    return {'hydra_cfg': hydra_cfg, 'run': run_params}


def set_up_experiment_run(hydra_cfg: DictConfig,
                          args: dict,
                          log_level: str = 'DEBUG') -> dict:

    # Add the input arguments as an extra subdict to the config
    hyperparam_name = hydra_cfg['NAME']
    run_params = put_to_dict({}, args, 'ARGS')

    # Setup the computing environment *## cfg['run']['MACHINE']
    machine_dict = set_up_environment(machine_config=cfg_key(hydra_cfg, 'config', 'MACHINE'),
                                      local_rank=args['local_rank'])
    run_params = put_to_dict(run_params, machine_dict, 'MACHINE')

    # If you do not want to mount the artifacts directory, and find easier just to "aws sync" at the end
    if args['s3_mount']:
        raise NotImplementedError('The real-time write to S3 seems like a dream at the moment, so aws sync at the end')

    try:
        debug_mounts(mounts_list=get_mounts_from_args(args=args))
    except Exception as e:
        logger.error('Failed to run the mount tests, e = {}'.format(e))
        raise IOError('Failed to run the mount tests, e = {}'.format(e))

    # Set-up run parameters
    run_params['PARAMS'] = set_up_run_params(hydra_cfg=hydra_cfg,
                                             args=args,
                                             hyperparam_name=hyperparam_name)

    # Define hyperparameters for Experimnent Tracking (MLflow/WANDB)
    run_params['HYPERPARAMETERS'],run_params['HYPERPARAMETERS_FLAT'] = (
        define_hyperparams_from_config(hydra_cfg=hydra_cfg))

    # Add hash to names to make them unique
    if cfg_key(hydra_cfg, 'config', 'LOGGING', 'unique_hyperparam_name_with_hash'):
        run_params['PARAMS'] = use_dict_hash_in_names(params=cfg_key(run_params, 'PARAMS'),
                                                      config_hash=cfg_key(run_params, 'PARAMS', 'config_hash'))

    # Set-up the log files (loguru, stdout/stderr)
    run_params = set_up_log_files(run_params=run_params,
                                  hyperparam_name=hyperparam_name,
                                  log_level=log_level)

    # Add run/environment-dependent metadata (e.g. library versions, etc.)
    run_params['METADATA'] = get_run_metadata()

    # Initialize ML logging (experiment tracking)
    run_params['MLFLOW'], mlflow_dict = (
        init_mlflow_logging(hydra_cfg=hydra_cfg,
                            run_params=run_params,
                            mlflow_config=cfg_key(hydra_cfg, 'config', 'LOGGING', 'MLFLOW'),
                            experiment_name=args['project_name'],
                            run_name=cfg_key(run_params, 'PARAMS', 'hyperparam_name'),
                            server_uri=cfg_key(hydra_cfg, 'config', 'SERVICES', 'MLFLOW', 'server_URI')))

    # Convert to OmegaConf
    logger.debug('Convert run_params dict to OmegaConf DictConfig')
    run_params = OmegaConf.create(run_params)
    assert type(run_params) == DictConfig
    logger.debug('Done setting up the experiment run parameters')

    return run_params


def get_repo_base_dir(working_dir: str,
                      repo_name: str = 'minivess_mlops'):

    # quick and dirty recursion limit
    working_dir_orig = working_dir
    if os.path.split(working_dir)[-1] == repo_name:
        return working_dir
    else:
        working_dir = os.path.split(working_dir)[0]
        if os.path.split(working_dir)[-1] == repo_name:
            return working_dir
        else:
            working_dir = os.path.split(working_dir)[0]
            if os.path.split(working_dir)[-1] == repo_name:
                return working_dir
            else:
                logger.error('Could not find the repo base dir, from {}'.format(working_dir_orig))
                raise IOError('Could not find the repo base dir, from {}'.format(working_dir_orig))


def get_repo_dir(return_src: bool = False,
                 repo_name: str = 'minivess_mlops') -> str:

    cwd = os.getcwd()
    logger.debug('Trying to find the repo dir from cwd = "{}"'.format(cwd))
    repo_dir = get_repo_base_dir(working_dir=cwd,
                                 repo_name=repo_name)
    logger.info('Git base repo found in "{}"'.format(repo_dir))

    if return_src:
        repo_dir = os.path.join(repo_dir, 'src')

    return repo_dir


def set_up_run_params(hydra_cfg: DictConfig,
                      args: dict,
                      hyperparam_name: str):

    # Create hash from config dict
    config_hash = dict_hash(dictionary=deepcopy(hydra_cfg['config']),
                            drop_keys = ('SERVICES', ))
    start_time = get_datetime_string()

    output_experiments_base_dir = os.path.join(args['output_dir'], 'experiments')
    logger.info('Save the run-specific parameters to cfg["run"]["PARAMS"]')
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

    run_params['requirements-txt_path'] = get_requirements_txt_path(repo_dir=run_params['repo_dir'])

    # Init variables for 'run'
    run_params['repeat_artifacts'] = {}
    run_params['ensemble_artifacts'] = {}
    run_params['fold_dir'] = {}
    print_dict_to_logger(dict_in=run_params, prefix='')

    return run_params


def get_requirements_txt_path(repo_dir: str,
                              raise_error: bool = True) -> str:

    # Get the requirements.txt path for MLflow reproduction, which will be auto-generated by the Docker
    # from the Poetry when running this from Docker (Dockerfile_env), when running locally without Docker,
    # there should be a manually generated requirements.txt in the "deployment" folder
    # i.e. "poetry export --format requirements.txt --without-hashes --output requirements.txt"
    if is_docker():
        # "Hard-coded in Dcokerfile_env"
        requirements_path = os.path.join('/home', 'minivessuser', 'requirements.txt')
        logger.debug('Code is run inside Docker')
    else:
        logger.debug('Code is run outside Docker')
        requirements_path = os.path.join(repo_dir, 'deployment', 'requirements.txt')

    if not os.path.exists(requirements_path):
        requirements_path = None
        logger.error('Could not find requirements.txt from "{}"\n'
                     'Cannot reproduce this run in MLflow'.format(requirements_path))
        if raise_error:
            raise IOError('Could not find requirements.txt from "{}"\n'
                          'Cannot reproduce this run in MLflow'.format(requirements_path))
        else:
            logger.info('Setting the requirements.txt path to None! \n'
                        'Causes issues to MLflow reproduction, but the training runs just fine')
    else:
        logger.info('"requirements.txt" found from "{}"'.format(requirements_path))

    return requirements_path


def use_dict_hash_in_names(params: dict,
                           config_hash: str) -> dict:

    params['hyperparam_name'] += '_{}'.format(config_hash)
    params['output_experiment_dir'] += '_{}'.format(config_hash)
    logger.info('Unique hyperparam name with hash')
    logger.info('  hyperparam_name = "{}"'.format(params['hyperparam_name']))
    logger.info('  output_experiment_dir = "{}"'.format(params['output_experiment_dir']))

    return params


def define_hyperparams_from_config(hydra_cfg: DictConfig):

    # Get a predefined smaller subset to be logged as MLflow/WANDB columns/hyperparameters
    # to make the dashboards cleaner, or alternatively you can just dump the whole config['config']
    hyperparameters = define_hyperparam_run_params(hydra_cfg)

    # TODO! Fix this with the updated nesting, with some nicer recursive function for undefined depth
    hyperparameters_flat = {'placeholder_key': 'value'}  # flatten_nested_dictionary(dict_in=config['hyperparameters'])
    logger.info('Save the derived hyperparameters to config["hyperparameters"]')
    logger.info(hyperparameters)  # TODO add to print_the_dict_to_logger() nested dicts as well

    return hyperparameters, hyperparameters_flat


def set_up_log_files(run_params: dict,
                     hyperparam_name: str,
                     log_level: str = 'DEBUG'):

    log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | "
                  "<yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>")
    log_file = 'log_{}.txt'.format(hyperparam_name)
    run_params['PARAMS']['output_log_path'] = os.path.join(run_params['PARAMS']['output_experiment_dir'], log_file)
    try:
        logger.add(run_params['PARAMS']['output_log_path'],
                   level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)
    except Exception as e:
        logger.error('Problem initializing the log file to the artifacts output, permission issues?? e = {}'.format(e))
        raise IOError('Problem initializing the log file to the artifacts output, have you created one? '
                      'do you have the permissions correct? See README.md for the "minivess_mlops_artifacts" creation'
                      'with symlink to /mnt \n error msg = {}'.format(e))
    logger.info('Log (loguru) will be saved to disk to "{}"'.format(run_params['PARAMS']['output_log_path']))

    log_file = 'stdout_{}.txt'.format(hyperparam_name)
    run_params['PARAMS']['stdout_log_path'] = os.path.join(run_params['PARAMS']['output_experiment_dir'], log_file)
    logger.info('Stdout will be saved to disk to "{}"'.format(run_params['PARAMS']['stdout_log_path']))
    sys.stdout = open(run_params['PARAMS']['stdout_log_path'], 'w')
    sys.stderr = sys.stdout

    return run_params


def set_up_environment(machine_config: DictConfig,
                       local_rank: int):

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


def dict_hash(dictionary: Dict[str, Any],
              drop_keys: tuple) -> str:
    """
    MD5 hash of a dictionary.
    https://www.doc.ic.ac.uk/~nuric/coding/how-to-hash-a-dictionary-in-python.html
    """

    logger.info('Computing hash for the Hydra config dictionary')
    dictionary: dict = OmegaConf.to_container(dictionary)

    # The idea was to have the hash to track the uniqueness of the dictionary (see if you are running the
    # experiment with exactly the same configs), so you don't want to have any stochastic keys here (random seeds),
    # or anything that is irrelevant for the training but varies across users, organisations, teams etc.
    for drop_key in drop_keys:
        dictionary.pop(drop_key, None)
        logger.debug('Dropping "{}" from the dictionary before computing hash'.format(drop_key))

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


def define_hyperparam_run_params(hydra_cfg: DictConfig) -> dict:
    """
    To be optimized later? You could read these from the config as well, which subdicts count as
    experiment hyperparameters. Not just dumping all possibly settings that maybe not have much impact
    on the "ML science" itself, e.g. number of workers for dataloaders or something
    :param config:
    :return:
    """

    hyperparams = {}
    cfg = hydra_cfg['config']
    logger.info('Hand-picking the keys/subdicts from "config" that are logged as hyperparameters for MLflow/WANDB')

    # What datasets were you used for training the model
    hyperparams['datasets'] = cfg_key(cfg, 'DATA', 'DATA_SOURCE', 'DATASET_NAMES')

    # What model and architecture hyperparams you used
    hyperparams['models'] = {}
    model_names = cfg_key(cfg, 'MODEL', 'META_MODEL')
    for model_name in model_names:
        # hyperparams['models'][model_name] = {'architecture': cfg_key(cfg, 'MODEL', model_name)}
        hyperparams = put_to_dict(hyperparams,
                                  {'architecture': cfg_key(cfg, 'MODEL', model_name)},
                                  'models', model_name)

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


