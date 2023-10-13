import argparse
import os

import sys

from src.utils.dataloder_utils import define_dataset_and_dataloader

src_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(src_path)[0]
sys.path.insert(0, project_path) # so that src. is imported correctly also in VSCode by default

from src.train import training_script
from src.utils.config_utils import import_config
from src.utils.data_utils import import_datasets

import warnings

from src.utils.metadata_utils import get_run_metadata

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
# __init__.py:127: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage
# will be the only storage class. This should only matter to you if you are using storages directly.
# To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()

# TOEXPLORE! MLflow uses pydantic and it throws: Field "model_server_url" has conflict with protected namespace "model_".
#  Field "model_server_url" has conflict with protected namespace "model_".

# Control logger level here, make it nicer later, include the level as an input argument to the script
# https://github.com/Delgan/loguru/issues/138#issuecomment-1491571574
from loguru import logger
LOG_MIN_LEVEL = "DEBUG"

def my_filter(record):
    return record["level"].no >= logger.level(LOG_MIN_LEVEL).no
logger.remove()
logger.add(sys.stderr, filter=my_filter)
# LOG_MIN_LEVEL = "DEBUG"

# TODO! Set this flag based on the run_mode when you actually implement it for CI/CD
SKIP_TRAINING = False


def parse_args_to_dict():
    parser = argparse.ArgumentParser(description='Segmentation pipeline for Minivess dataset')
    parser.add_argument('-c', '--task_config-file', type=str, required=True,
                        default='task_config.yaml',
                        help="Name of your task-specific .yaml file, e.g. 'config_test'")
    parser.add_argument('-dbg', '--debug_mode', action="store_const", const=True,
                        help="Sets debug flag on. Quick way for example to train for less epochs or something else,"
                             "when you are not actually training but mostly developing the code")
    parser.add_argument('-data', '--data_dir', type=str, required=False,
                        default='/mnt/minivess_mlops_artifacts/data',
                        help="Where the data is downloaded, or what dir needs to be mounted when you run this"
                             "on Docker")
    parser.add_argument('-output', '--output_dir', type=str, required=False,
                        default='/mnt/minivess_mlops_artifacts/output',
                        help="Where the data is downloaded, or what dir needs to be mounted when you run this"
                             "on Docker")
    parser.add_argument('-rank', '--local_rank', type=int, required=False, default=0,
                        help="node rank for distributed training")
    parser.add_argument('-p', '--project_name', type=str, required=False,
                        default='MINIVESS_segmentation_TEST',
                        help="Name of the project in WANDB/MLOps. Keep the same name for all the segmentation"
                             "experiments so that you can compare how tweaks affect segmentation performance."
                             "Obviously create a new project if you have MINIVESS_v2 or some other dataset, when"
                             "you cannot meaningfully compare e.g. DICE score from dataset 1 to dataset 2")
    # TODO! Add "run_mode" or something: ['train', 'test_dataload', 'test_train', 'debug]
    #  to allow some CI/CD
    return vars(parser.parse_args())


if __name__ == '__main__':

    # TOADD! Actual hyperparameter config that defines the experiment to run
    hyperparam_runs = {'hyperparam_example_name'}
    hparam_run_results = {}
    for hyperparam_idx, hyperparam_name in enumerate(hyperparam_runs):

        # Import the config
        args = parse_args_to_dict()
        config = import_config(args, task_config_file = args['task_config_file'],
                               hyperparam_name=hyperparam_name, log_level=LOG_MIN_LEVEL)

        # Add run/environment-dependent metadata (e.g. library versions, etc.)
        config['metadata'] = get_run_metadata()

        # Collect the data and define splits
        fold_split_file_dicts, config['config']['DATA'] = \
            import_datasets(data_config=config['config']['DATA'],
                            data_dir=args['data_dir'],
                            debug_mode=args['debug_mode'],
                            config=config)

        # Create and validate datasets and dataloaders
        experim_datasets, experim_dataloaders = \
            define_dataset_and_dataloader(config=config,
                                          fold_split_file_dicts=fold_split_file_dicts)

        # Train for n folds, n repeats, n epochs (single model)
        if SKIP_TRAINING:
            logger.info('Skipping the training for now, hyperparam_name = {}'
                        'e.g. when running CI/CD tasks for data checks'.format(hyperparam_name))
        else:
            hparam_run_results[hyperparam_name] = \
                training_script(hyperparam_name=hyperparam_name,
                                experim_dataloaders=experim_dataloaders,
                                config=config,
                                machine_config=config['config']['MACHINE'],
                                output_dir=config['run']['output_experiment_dir'])

    logger.info('Done with the experiment!')
