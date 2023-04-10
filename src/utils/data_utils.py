import os
from loguru import logger

from src.datasets.minivess import download_and_extract_minivess_dataset, define_minivess_dataset, define_minivess_splits
from src.utils.transforms import define_transforms


def define_dataset_and_dataloader(config: dict, dataset_dir: str):

    datasets = define_datasets(config, dataset_dir)


def define_datasets(config: dict, dataset_dir: str):

    transforms = define_transforms(dataset_config=config['config']['DATA']['DATASET'])
    split_file_dicts = define_minivess_splits(dataset_dir=dataset_dir, data_splits_config=config['config']['DATA']['SPLITS'])

    if config['config']['DATA']['DATA_SOURCE']['DATASET_NAME'] == 'minivess':
        datasets = define_minivess_dataset(dataset_config=config['config']['DATA']['DATASET'],
                                           dataset_dir=dataset_dir,
                                           split_file_dicts=split_file_dicts,
                                           transforms=transforms)
    else:
        raise NotImplementedError('Only implemented minivess dataset now!, '
                                  'not = "{}"'.format(config['config']['DATA']['DATASET_NAME']))

    return datasets


def import_dataset(data_config, data_dir):

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        logger.info('Data directory did not exist in "{}", creating it', data_dir)

    # TOADD! how to get the filename and extension automagically when in the link you don't have this info?
    input_url = data_config['DATA_SOURCE']['DATA_DOWNLOAD_URL']
    dataset_name = data_config['DATA_SOURCE']['DATASET_NAME']

    if dataset_name == 'minivess':
        dataset_dir = download_and_extract_minivess_dataset(input_url=input_url, data_dir=data_dir)
    else:
        raise NotImplementedError('Do not yet know how to download a dataset '
                                  'called = "{}"'.format(dataset_name))

    return dataset_dir


def get_dir_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size