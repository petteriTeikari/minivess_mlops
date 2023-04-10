import argparse

from src.utils.config_utils import import_config
from src.utils.data_utils import import_dataset, define_dataset_and_dataloader


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
    return vars(parser.parse_args())


if __name__ == '__main__':

    args = parse_args_to_dict()
    config = import_config(args, task_config_file = args['task_config_file'])
    dataset_dir = import_dataset(data_config=config['config']['DATA'], data_dir=args['data_dir'])
    datasets, dataloaders = define_dataset_and_dataloader(config, dataset_dir = dataset_dir)

