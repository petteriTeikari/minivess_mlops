import argparse
import os
import sys

from src.train import training_script
from src.utils.config_utils import import_config, set_up_environment
from src.utils.data_utils import define_dataset_and_dataloader, import_datasets

import warnings

from src.utils.metadata_utils import get_run_metadata

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
# __init__.py:127: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage
# will be the only storage class. This should only matter to you if you are using storages directly.
# To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()

# Control logger level here, make it nicer later
# https://github.com/Delgan/loguru/issues/138#issuecomment-1491571574
from loguru import logger
min_level = "INFO"

def my_filter(record):
    return record["level"].no >= logger.level(min_level).no
logger.remove()
logger.add(sys.stderr, filter=my_filter)
min_level = "DEBUG"


def parse_args_to_dict():
    parser = argparse.ArgumentParser(description='Segmentation pipeline for Minivess dataset')
    parser.add_argument('-c', '--task_config-file', type=str, required=True,
                        default='task_config.yaml',
                        help="Name of your task-specific .yaml file, e.g. 'config_test'")
    parser.add_argument('-dbg', '--debug_mode', action="store_const", const=False,
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
                        default='MINIVESS_segmentation',
                        help="Name of the project in WANDB/MLOps. Keep the same name for all the segmentation"
                             "experiments so that you can compare how tweaks affect segmentation performance."
                             "Obviously create a new project if you have MINIVESS_v2 or some other dataset, when"
                             "you cannot meaningfully compare e.g. DICE score from dataset 1 to dataset 2")
    return vars(parser.parse_args())


if __name__ == '__main__':

    # TOADD! Actual hyperparameter config that defines the experiment to run
    hyperparam_runs = {'hparam_tag'}
    hparam_run_results = {}
    for hyperparam_idx, hyperparam_name in enumerate(hyperparam_runs):

        # Import the config
        args = parse_args_to_dict()
        config = import_config(args, task_config_file = args['task_config_file'])

        # Add run/environment-dependent metadata (e.g. library versions, etc.)
        config['metadata'] = get_run_metadata()

        # Collect the data and define splits
        fold_split_file_dicts, config['config']['DATA'] = \
            import_datasets(data_config=config['config']['DATA'], data_dir=args['data_dir'])

        # Create and validate datasets and dataloaders
        experim_datasets, experim_dataloaders = \
            define_dataset_and_dataloader(config, fold_split_file_dicts=fold_split_file_dicts)

        # Train for n folds, n repeats, n epochs (single model)
        hparam_run_results[hyperparam_name] = \
            training_script(experim_dataloaders=experim_dataloaders,
                            config=config,
                            training_config=config['config']['TRAINING'],
                            model_config=config['config']['MODEL'],
                            machine_config=config['config']['MACHINE'],
                            output_dir=os.path.join(config['ARGS']['output_dir'], hyperparam_name))

        a = 1

