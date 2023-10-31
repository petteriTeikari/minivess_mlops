import argparse
import os
import sys
import time

from src.log_ML.mlflow_admin import mlflow_update_best_model
from src.training.experiment import train_run_per_hyperparameters
from src.utils.general_utils import print_dict_to_logger

from loguru import logger
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
# __init__.py:127: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage
# will be the only storage class. This should only matter to you if you are using storages directly.
# To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()

src_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(src_path)[0]
sys.path.insert(0, project_path)  # so that src. is imported correctly also in VSCode by default

# TOEXPLORE! MLflow uses pydantic and it throws: Field "model_server_url"
# has conflict with protected namespace "model_".
#  Field "model_server_url" has conflict with protected namespace "model_".

# Control logger level here, make it nicer later, include the level as an input argument to the script
# https://github.com/Delgan/loguru/issues/138#issuecomment-1491571574
LOG_MIN_LEVEL = "DEBUG"


def my_filter(record):
    return record["level"].no >= logger.level(LOG_MIN_LEVEL).no


logger.remove()
logger.add(sys.stderr, filter=my_filter)
# LOG_MIN_LEVEL = "DEBUG"


def parse_args_to_dict():
    parser = argparse.ArgumentParser(description='Segmentation pipeline for Minivess dataset')
    parser.add_argument('-c', '--task_config-file', type=str, required=True,
                        default='train_task_test.yaml',
                        help="Name of your task-specific .yaml file, e.g. 'config_test'")
    # TODO! base_config
    parser.add_argument('-run', '--run_mode', type=str, required=False,
                        default='train', choices=['train', 'debug',
                                                  'test_data', 'test_dataload', 'test_train'],
                        help="Use 'train' for the actual training, 'test_xxx' are meant for CI/CD tasks,"
                             "and debug is for development with very tiny subsets of data")
    parser.add_argument('-data', '--data_dir', type=str, required=False,
                        default='/mnt/minivess-dvc-cache',
                        help="Where the data is downloaded, or what dir needs to be mounted when you run this"
                             "on Docker")
    parser.add_argument('-output', '--output_dir', type=str, required=False,
                        default='/mnt/minivess-artifacts',
                        help="Where the data is downloaded, or what dir needs to be mounted when you run this"
                             "on Docker")
    parser.add_argument('-s3', '--s3_mount', action='store_true',
                        help="If you have issues with the mounting (mountpoint-s3, s3fs, etc.), "
                             "and are okay with just aws sync at the end of the run")
    parser.add_argument('-no-s3', '--no-s3_mount', dest='s3_mount', action='store_false')
    parser.set_defaults(s3_mount=False)
    parser.add_argument('-rank', '--local_rank', type=int, required=False, default=0,
                        help="node rank for distributed training")
    # TODO! log_level
    parser.add_argument('-p', '--project_name', type=str, required=False,
                        default='minivess-test2',
                        help="Name of the project in WANDB/MLOps. Keep the same name for all the segmentation"
                             "experiments so that you can compare how tweaks affect segmentation performance."
                             "Obviously create a new project if you have MINIVESS_v2 or some other dataset, when"
                             "you cannot meaningfully compare e.g. DICE score from dataset 1 to dataset 2")
    args_dict = vars(parser.parse_args())
    logger.info('Parsed input arguments:')
    print_dict_to_logger(args_dict, prefix='')

    return args_dict


if __name__ == '__main__':

    # Placeholder for hyperparameter sweep, iterate through these
    hyperparam_runs = {'hyperparam_example_name'}
    hparam_run_results = {}
    t0 = time.time()
    args = parse_args_to_dict()

    # These are all parallel jobs (in theory), e.g. if you had 100 hyperparam combinations,
    # you could spin 100 instances here and have your results in 100x the time
    for hyperparam_idx, hyperparam_name in enumerate(hyperparam_runs):
        logger.info('Starting hyperparam run {}/{}: {}'.format(hyperparam_idx + 1,
                                                               len(hyperparam_runs),
                                                               hyperparam_name))
        hparam_run_results[hyperparam_name] = train_run_per_hyperparameters(args)

    # Update the best MLflow model
    mlflow_update_best_model(project_name=args['project_name'])

    logger.info('Done in {:.0f} seconds with the execution!'.format(time.time() - t0))
