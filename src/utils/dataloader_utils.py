from loguru import logger
from monai.data import DataLoader, list_data_collate, Dataset
from omegaconf import DictConfig

from src.datasets.torch_dataset import define_torch_dataset, get_sample_keys
from src.utils.config_utils import cfg_key
from tests.data.dataloader_tests import ml_test_dataloader_dict_integrity
from tests.data.dataset_tests import ml_test_dataset_summary
from src.datasets.minivess import define_minivess_dataset
from src.utils.transforms import define_transforms, no_aug


def define_datasets(cfg: dict,
                    fold_split_file_dicts: dict,
                    debug_testing: bool = False):
    """
    Each of the validation and test split now contain dicts
        # * If you would like to for example track the model improvement using a multiple validation splits, e.g.
        #   Dice was best at epoch 23 for Dataset A, and at epoch 55 for Dataset B, and you have 2 different
        #   "best models" saved to disk
        # * Similarly you would like to have multiple external test splits, and you would like to see
        #   how differently the model learns the different datasets? The model does not generalize very well for
        #   specific source of data (which would be masked if you just grouped all the test sampled in one dataloader?
    """

    if debug_testing:
        logger.warning('WARNING! Debugging mode on for dataset, and errors are intentionally added!')

    datasets, ml_test_dataset = {}, {}
    logger.info('Creating (MONAI/PyTorch) Datasets')
    for f, fold_name in enumerate(fold_split_file_dicts):
        datasets[fold_name], ml_test_dataset[fold_name] = {}, {}

        for i, dataset_name in enumerate(fold_split_file_dicts[fold_name].keys()):
            logger.info('Creating (MONAI/PyTorch) Datasets for dataset source = "{}"', dataset_name)
            dataset_config = cfg_key(cfg, 'hydra_cfg', 'config', 'DATA', 'DATA_SOURCE', dataset_name)
            split_file_dicts = fold_split_file_dicts[fold_name][dataset_name]

            # FIXME! What if you want to do some diverse inference with a different set of augmentation
            #  per each fold or for each repeat? See e.g. https://arxiv.org/abs/2007.04206
            #  You do not necessarily need to save these to a dict, but you would to define it here
            #  based on repeat or/and fold-wise
            transforms = define_transforms(dataset_config=dataset_config,
                                           transform_config_per_dataset=cfg_key(dataset_config, 'TRANSFORMS'),
                                           transform_config=cfg_key(cfg, 'hydra_cfg', 'config', 'TRANSFORMS'),
                                           device=cfg_key(cfg, 'run', 'MACHINE', 'device'),
                                           keys_in_samples=get_sample_keys(split_file_dicts))

            datasets[fold_name][dataset_name], ml_test_dataset[fold_name][dataset_name] = \
                define_torch_dataset(dataset_name=dataset_name,
                                     dataset_config=dataset_config,
                                     split_file_dicts=split_file_dicts,
                                     transforms=transforms,
                                     debug_testing=debug_testing)

    # Test report for dataset integrity
    all_tests_ok, report_string = ml_test_dataset_summary(ml_test_dataset)

    if not all_tests_ok:
        print(report_string)
        logger.debug('Your chance to write the dataset ML test "report_to_string" to a file or something')
        logger.error('Dataset contains illegal types!')
        raise TypeError('Dataset contains illegal types!')

    return datasets


def redefine_dataloader_for_inference(dataloader_batched,
                                      cfg: dict,
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
            raise NotImplementedError('Only implemented minivess dataset now!, not = "{}"'.
                                      format(cfg_key(cfg, 'hydra_cfg', 'config', 'DATA', 'DATASET_NAME')))

        return dataset

    dataset = redefine_dataset_for_inference(dataloader_batched, dataset_name, device)

    dataloader_config = cfg_key(cfg, 'hydra_cfg', 'config', 'DATA', 'DATALOADER')
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            num_workers=cfg_key(dataloader_config, split, 'NUM_WORKERS'),
                            collate_fn=list_data_collate)

    logger.info('Redefining MONAI dataset/dataloader for inference (batch size = 1, original resolution), '
                'no_samples = {}'.format(len(dataloader)))

    return dataloader


def define_dataset_and_dataloader(cfg: dict,
                                  fold_split_file_dicts: dict):

    datasets = (
        define_datasets(cfg=cfg,
                        fold_split_file_dicts=fold_split_file_dicts,
                        debug_testing=cfg_key(cfg, 'hydra_cfg', 'config', 'TESTING', 'DATASET', 'debug_testing')))

    dataloaders = define_dataloaders(datasets=datasets,
                                     cfg=cfg,
                                     dataloader_config=cfg_key(cfg, 'hydra_cfg', 'config', 'DATA', 'DATALOADER'))

    # FIXME remove this eventually, and do not do quick and dirty stuff :)
    datasets, dataloaders = quick_and_dirty_training_dataloader_creation(datasets, dataloaders)
    logger.info('Done with the training preparation\n')

    # Test that you can use the constructed dataloaders actually
    if cfg_key(cfg, 'hydra_cfg', 'config', 'TESTING', 'DATALOADER', 'DATA_VALIDITY', 'enable'):
        ml_test_dataloader_dict_integrity(dataloaders,
                                          test_config=cfg_key(cfg, 'hydra_cfg', 'config', 'TESTING', 'DATALOADER'),
                                          run_params=cfg_key(cfg, 'run', 'PARAMS'),
                                          debug_testing=cfg_key(cfg, 'hydra_cfg', 'config', 'TESTING',
                                                                'DATALOADER', 'debug_testing'))
    else:
        logger.info('Skip ML tests for the dataloader integrity ("DATA VALIDITY")')
        # TO-OPTIMIZE the naming of these

    return datasets, dataloaders


def quick_and_dirty_training_dataloader_creation(datasets, dataloaders,
                                                 dataset_name: str = 'MINIVESS',
                                                 fold_name: str = 'fold0'):

    # FIXME: Now we cheat a bit and create this manually, update this when you actually have multiple
    #  datasets and you would like to create one training dataloader from all the datasets, and multiple
    #  dataloaders then in a dict for TEST and VAL splits
    if dataset_name in datasets[fold_name]:
        logger.info('Quick and dirty fixing of MINIVESS dataset to the desired format of 1 TRN dataloader, '
                    'and multiple VAL/TEST splits in dictionary')
        datasets[fold_name] = datasets[fold_name][dataset_name]

        dataloaders_out = {fold_name: {}}
        # this is now a single dictionary. Not sure if there is an use case where you would like to
        # simultaneously have two training dataloaders?
        dataloaders_out[fold_name]['TRAIN'] = dataloaders[fold_name][dataset_name]['TRAIN']
        # these are now subdictionaries, so you could use multiple validation sets to track model improvement
        # or for using multiple external test sets for generalization evaluation purposes
        dataloaders_out[fold_name]['VAL'] = {dataset_name: dataloaders[fold_name][dataset_name]['VAL']}
        dataloaders_out[fold_name]['TEST'] = {dataset_name: dataloaders[fold_name][dataset_name]['TEST']}
    else:
        logger.info('No quick asnd dirty fix for dataset_name = "{}"'.format(dataset_name))
        dataloaders_out = dataloaders

    return datasets, dataloaders_out


def define_dataloaders(datasets: dict,
                       cfg: dict,
                       dataloader_config: dict):

    logger.info('Creating (MONAI/PyTorch) Dataloaders')
    dataloaders = {}
    for f, fold_name in enumerate(datasets.keys()):
        dataloaders[fold_name] = {}
        for i, dataset_name in enumerate(datasets[fold_name].keys()):
            dataloaders[fold_name][dataset_name] = {}
            dataset_per_name = datasets[fold_name][dataset_name]
            for j, split in enumerate(dataset_per_name.keys()):
                logger.info('Dataloader for fold = {}, dataset = {}, split = {}, batch_sz = {}, num_workers = {}'.
                            format(fold_name, dataset_name, split,
                                   cfg_key(dataloader_config, split, 'BATCH_SZ'),
                                   cfg_key(dataloader_config, split, 'NUM_WORKERS')))
                dataloaders[fold_name][dataset_name][split] = \
                    DataLoader(dataset_per_name[split],
                               batch_size=cfg_key(dataloader_config, split, 'BATCH_SZ'),
                               num_workers=cfg_key(dataloader_config, split, 'NUM_WORKERS'),
                               collate_fn=list_data_collate)

    return dataloaders
