import argparse

import torch
from loguru import logger

from src.train import train_model
from src.utils.config_utils import import_config
from src.utils.data_utils import define_dataset_and_dataloader, import_datasets

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
# __init__.py:127: UserWarning: TypedStorage is deprecated. It will be removed in the future and UntypedStorage
# will be the only storage class. This should only matter to you if you are using storages directly.
# To access UntypedStorage directly, use tensor.untyped_storage() instead of tensor.storage()


def parse_args_to_dict():
    parser = argparse.ArgumentParser(description='Segmentation pipeline for Minivess dataset')
    parser.add_argument('-c', '--task_config-file', type=str, required=True, default='base_config.yaml',
                        help="Name of your task-specific .yaml file, e.g. 'config_test'")
    parser.add_argument('-dbg', '--debug_mode', action="store_const", const=False,
                        help="Sets debug flag on. Quick way for example to train for less epochs or something else,"
                             "when you are not actually training but mostly developing the code")
    parser.add_argument('-data', '--data_dir', type=str, required=True, default='/home/petteri/minivess_data',
                        help="Where the data is downloaded, or what dir needs to be mounted when you run this"
                             "on Docker")
    parser.add_argument('-rank', '--local_rank', type=int, required=False, default=0,
                        help="node rank for distributed training")
    return vars(parser.parse_args())


if __name__ == '__main__':

    args = parse_args_to_dict()
    config = import_config(args, task_config_file = args['task_config_file'])
    device = torch.device(f"cuda:{0}") # FIXME set up the hardware environment here already
    dataset_dirs = import_datasets(data_config=config['config']['DATA'], data_dir=args['data_dir'])
    datasets, dataloaders = define_dataset_and_dataloader(config, dataset_dirs=dataset_dirs, device=device)
    logger.info('Done with the training preparation\n')

    train_model(dataloaders=dataloaders,
                config=config,
                training_config=config['config']['TRAINING'],
                model_config=config['config']['MODEL'],
                machine_config=config['config']['MACHINE'],
                local_rank=args['local_rank'])

