import os
from loguru import logger

from src.datasets.minivess import download_and_extract_minivess_dataset, define_minivess_dataset, \
    define_minivess_splits, get_minivess_filelisting
from src.utils.general_utils import check_if_key_in_dict
from src.utils.transforms import define_transforms, no_aug

from monai.data import DataLoader, list_data_collate, Dataset

def define_dataset_and_dataloader(config: dict, fold_split_file_dicts: dict):

    datasets = define_datasets(config, fold_split_file_dicts=fold_split_file_dicts)
    dataloaders = define_dataloaders(datasets, config,
                                     dataloader_config=config['config']['DATA']['DATALOADER'])

    # FIXME remove this eventually, and do not do quick and dirty stuff :)
    datasets, dataloaders = quick_and_dirty_training_dataloader_creation(datasets, dataloaders)
    logger.info('Done with the training preparation\n')

    return datasets, dataloaders


def quick_and_dirty_training_dataloader_creation(datasets, dataloaders,
                                                 dataset_name: str = 'MINIVESS',
                                                 fold_name: str = 'fold0'):

    # FIXME: Now we cheat a bit and create this manually, update this when you actually have multiple
    #  datasets and you would like to create one training dataloader from all the datasets, and multiple
    #  dataloaders then in a dict for TEST and VAL splits
    logger.info('Quick and dirty fixing of MINIVESS dataset to the desired format of 1 TRN dataloader, '
                'and multiple VAL/TEST splits in dictionary')
    datasets[fold_name] = datasets[fold_name][dataset_name]

    dataloaders_out = {}
    dataloaders_out[fold_name] = {}
    # this is now a single dictionary. Not sure if there is an use case where you would like to
    # simultaneously have two training dataloaders?
    dataloaders_out[fold_name]['TRAIN'] = dataloaders[fold_name][dataset_name]['TRAIN']
    # these are now subdictionaries, so you could use multiple validation sets to track model improvement
    # or for using multiple external test sets for generalization evaluation purposes
    dataloaders_out[fold_name]['VAL'] = {dataset_name: dataloaders[fold_name][dataset_name]['VAL']}
    dataloaders_out[fold_name]['TEST'] = {dataset_name: dataloaders[fold_name][dataset_name]['TEST']}

    return datasets, dataloaders_out


def define_dataloaders(datasets: dict, config: dict, dataloader_config: dict):

    logger.info('Creating (MONAI/PyTorch) Dataloaders')
    dataloaders = {}
    for f, fold_name in enumerate(datasets.keys()):
        dataloaders[fold_name] = {}
        for i, dataset_name in enumerate(datasets[fold_name].keys()):
            dataloaders[fold_name][dataset_name] = {}
            dataset_per_name = datasets[fold_name][dataset_name]
            for j, split in enumerate(dataset_per_name.keys()):
                dataloaders[fold_name][dataset_name][split] = \
                    DataLoader(dataset_per_name[split],
                               batch_size=dataloader_config[split]['BATCH_SZ'],
                               num_workers=dataloader_config[split]['NUM_WORKERS'],
                               collate_fn=list_data_collate)

    return dataloaders


def define_datasets(config: dict, fold_split_file_dicts: dict):
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
    for f, fold_name in enumerate(fold_split_file_dicts):
        datasets[fold_name] = {}

        for i, dataset_name in enumerate(fold_split_file_dicts[fold_name].keys()):
            logger.info('Creating (MONAI/PyTorch) Datasets for dataset source = "{}"', dataset_name)
            dataset_config = config['config']['DATA']['DATA_SOURCE'][dataset_name]

            # FIXME! What if you want to do some diverse inference with a different set of augmentation
            #  per each fold or for each repeat? See e.g. https://arxiv.org/abs/2007.04206
            #  You do not necessarily need to save these to a dict, but you would to define it here
            #  based on repeat or/and fold-wise
            transforms = define_transforms(dataset_config=dataset_config,
                                           transform_config_per_dataset=dataset_config['TRANSFORMS'],
                                           transform_config=config['config']['TRANSFORMS'],
                                           device=config['config']['MACHINE']['IN_USE']['device'])

            if dataset_name == 'MINIVESS':
                split_file_dicts = fold_split_file_dicts[fold_name][dataset_name]
                datasets[fold_name][dataset_name] = \
                    define_minivess_dataset(dataset_config=dataset_config,
                                            split_file_dicts=split_file_dicts,
                                            transforms=transforms)
            else:
                raise NotImplementedError('Only implemented minivess dataset now!, '
                                          'not = "{}"'.format(config['config']['DATA']['DATASET_NAME']))

    return datasets

def redefine_dataloader_for_inference(dataloader_batched,
                                      config: dict,
                                      dataset_name: str = 'MINIVESS',
                                      split: str = 'VAL',
                                      device: str = 'cpu'):

    def redefine_dataset_for_inference(dataloader_batched, dataset_name: str, device: str):

        dataset_object = dataloader_batched.dataset
        split_file_dict = dataset_object.data
        transforms = no_aug(device, for_inference=True)

        if dataset_name == 'MINIVESS':
            dataset = Dataset(data=split_file_dict,
                         transform=transforms)
        else:
            raise NotImplementedError('Only implemented minivess dataset now!, '
                                      'not = "{}"'.format(config['config']['DATA']['DATASET_NAME']))

        return dataset

    dataset = redefine_dataset_for_inference(dataloader_batched, dataset_name, device)

    dataloader_config = config['config']['DATA']['DATALOADER']
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=dataloader_config[split]['NUM_WORKERS'],
                            collate_fn=list_data_collate)

    logger.info('Redefining MONAI dataset/dataloader for inference (batch size = 1, original resolution), '
                'no_samples = {}'.format(len(dataloader)))

    return dataloader


def import_datasets(data_config: dict, data_dir: str):

    def reverse_fold_and_dataset_order(fold_split_file_dicts):
        # Note! if you combine multiple datasets, we assume that all the different datasets have similar folds
        #       i.e. enforce this splitting from the config. TOADD later
        dict_out = {}
        dataset_names = list(fold_split_file_dicts.keys())
        fold_names = list(fold_split_file_dicts[dataset_names[0]].keys())
        for fold_name in fold_names:
            dict_out[fold_name] = {}
            for dataset_name in dataset_names:
                dict_out[fold_name][dataset_name] = fold_split_file_dicts[dataset_name][fold_name]
        return dict_out


    datasets_to_import = data_config['DATA_SOURCE']['DATASET_NAME']
    logger.info('Importing the following datasets: {}', datasets_to_import)
    dataset_filelistings, fold_split_file_dicts = {}, {}
    for i, dataset_name in enumerate(datasets_to_import):
        dataset_filelistings[dataset_name], fold_split_file_dicts[dataset_name], data_config = \
            import_dataset(data_config, data_dir, dataset_name)

    # reverse fold and dataset_name in the fold_splits for easier processing afterwards
    fold_split_file_dicts = reverse_fold_and_dataset_order(fold_split_file_dicts)

    return fold_split_file_dicts, data_config


def import_dataset(data_config: dict, data_dir: str, dataset_name: str):

    logger.info('Importing: {}', dataset_name)
    dataset_cfg = data_config['DATA_SOURCE'][dataset_name]

    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        logger.info('Data directory did not exist in "{}", creating it', data_dir)

    if not check_if_key_in_dict(data_config['DATA_SOURCE'], dataset_name):
        raise IOError('You wanted to use the dataset = "{}", but you had not defined that in your config!\n'
                      'You should have something defined for this in config["config"]["DATA"], '
                      'see MINIVESS definition for an example'.format(dataset_name))

    if dataset_name == 'MINIVESS':

        input_url = dataset_cfg['DATA_DOWNLOAD_URL']
        dataset_dir = download_and_extract_minivess_dataset(input_url=input_url, data_dir=data_dir)
        filelisting, data_config['DATA_SOURCE'][dataset_name]['STATS'] = get_minivess_filelisting(dataset_dir)
        fold_split_file_dicts = define_minivess_splits(filelisting, data_splits_config=dataset_cfg['SPLITS'])
    else:
        raise NotImplementedError('Do not yet know how to download a dataset '
                                  'called = "{}"'.format(dataset_name))

    return filelisting, fold_split_file_dicts, data_config


def get_dir_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size