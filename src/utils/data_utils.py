import os
from loguru import logger

from src.datasets.minivess import download_and_extract_minivess_dataset, define_minivess_dataset, define_minivess_splits
from src.utils.general_utils import check_if_key_in_dict
from src.utils.transforms import define_transforms

from monai.data import DataLoader, list_data_collate

def define_dataset_and_dataloader(config: dict, dataset_dirs: dict, device):

    no_of_dataset_dirs = len(dataset_dirs)
    datasets = define_datasets(config, dataset_dirs, device)
    dataloaders = define_dataloaders(datasets, config,
                                     dataloader_config = config['config']['DATA']['DATALOADER'])
    datasets, dataloaders = quick_and_dirty_training_dataloader_creation(datasets, dataloaders)

    return datasets, dataloaders


def quick_and_dirty_training_dataloader_creation(datasets, dataloaders,
                                                 dataset_name: str = 'MINIVESS'):

    # FIXME: Now we cheat a bit and create this manually, update this when you actually have multiple
    #  datasets and you would like to create one training dataloader from all the datasets, and multiple
    #  dataloaders then in a dict for TEST and VAL splits
    logger.info('Quick and dirty fixing of MINIVESS dataset to the desired format of 1 TRN dataloader, '
                'and multiple VAL/TEST splits in dictionary')
    datasets = datasets[dataset_name]

    dataloaders_out = {}
    # this is now a single dictionary. Not sure if there is an use case where you would like to
    # simultaneously have two training dataloaders?
    dataloaders_out['TRAIN'] = dataloaders[dataset_name]['TRAIN']
    # these are now subdictionaries, so you could use multiple validation sets to track model improvement
    # or for using multiple external test sets for generalization evaluation purposes
    dataloaders_out['VAL'] = {dataset_name: dataloaders[dataset_name]['VAL']}
    dataloaders_out['TEST'] = {dataset_name: dataloaders[dataset_name]['TEST']}

    return datasets, dataloaders_out


def define_dataloaders(datasets: dict, config: dict, dataloader_config: dict):

    logger.info('Creating (MONAI/PyTorch) Dataloaders')
    dataloaders = {}
    for i, dataset_name in enumerate(datasets.keys()):
        dataloaders[dataset_name] = {}
        dataset_per_name = datasets[dataset_name]
        for j, split in enumerate(dataset_per_name.keys()):
            dataloaders[dataset_name][split] = \
                DataLoader(dataset_per_name[split],
                           batch_size=dataloader_config[split]['BATCH_SZ'],
                           num_workers=dataloader_config[split]['NUM_WORKERS'],
                           collate_fn=list_data_collate)

    return dataloaders


def define_datasets(config: dict, dataset_dirs: dict, device):
    """
    :param config: dict
        as imported form the task .yaml
    :param dataset_dirs: dict
        dictionary containing the directories associated with each of the dataset that you want to use
    :return:

    Each of the validation and test split now contain dicts
        # * If you would like to for example track the model improvement using a multiple validation splits, e.g.
        #   Dice was best at epoch 23 for Dataset A, and at epoch 55 for Dataset B, and you have 2 different
        #   "best models" saved to disk
        # * Similarly you would like to have multiple external test splits, and you would like to see
        #   how differently the model learns the different datasets? The model does not generalize very well for
        #   specific source of data (which would be masked if you just grouped all the test sampled in one dataloader?
    """

    datasets = {}
    logger.info('Creating (MONAI/PyTorch) Datasets')
    for i, dataset_name in enumerate(dataset_dirs.keys()):

        logger.info('Creating (MONAI/PyTorch) Datasets for dataset source = "{}"', dataset_name)
        dataset_dir = dataset_dirs[dataset_name]
        dataset_config = config['config']['DATA']['DATA_SOURCE'][dataset_name]
        transforms = define_transforms(dataset_config=dataset_config,
                                       transform_config_per_dataset=dataset_config['TRANSFORMS'],
                                       transform_config=config['config']['TRANSFORMS'],
                                       device=device)
        split_file_dicts = define_minivess_splits(dataset_dir=dataset_dir,
                                                  data_splits_config=dataset_config['SPLITS'])

        if dataset_name == 'MINIVESS':
            datasets[dataset_name] = \
                define_minivess_dataset(dataset_config=dataset_config,
                                        split_file_dicts=split_file_dicts,
                                        transforms=transforms)
        else:
            raise NotImplementedError('Only implemented minivess dataset now!, '
                                      'not = "{}"'.format(config['config']['DATA']['DATASET_NAME']))

    return datasets


def import_datasets(data_config: dict, data_dir: str):

    datasets_to_import = data_config['DATA_SOURCE']['DATASET_NAME']
    logger.info('Importing the following datasets: {}', datasets_to_import)
    dataset_dirs = {}
    for i, dataset_name in enumerate(datasets_to_import):
        dataset_dirs[dataset_name] = import_dataset(data_config, data_dir, dataset_name)

    return dataset_dirs


def import_dataset(data_config: dict, data_dir: str, dataset_name: str):

    logger.info('Importing: {}', dataset_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        logger.info('Data directory did not exist in "{}", creating it', data_dir)

    if not check_if_key_in_dict(data_config['DATA_SOURCE'], dataset_name):
        raise IOError('You wanted to use the dataset = "{}", but you had not defined that in your config!\n'
                      'You should have something defined for this in config["config"]["DATA"], '
                      'see MINIVESS definition for an example'.format(dataset_name))

    if dataset_name == 'MINIVESS':
        input_url = data_config['DATA_SOURCE'][dataset_name]['DATA_DOWNLOAD_URL']
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