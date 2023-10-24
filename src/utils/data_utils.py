import os
from loguru import logger
from omegaconf import DictConfig

from src.datasets.minivess import import_minivess_dataset
from src.log_ML.mlflow_log import mlflow_log_dataset
from src.utils.general_utils import check_if_key_in_dict


def import_datasets(data_config: DictConfig,
                    data_dir: str,
                    config: DictConfig,
                    exp_run: dict,
                    run_mode: str = 'train'):

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

    datasets_to_import = data_config['DATA_SOURCE']['DATASET_NAMES']
    logger.info('Importing the following datasets: {}', datasets_to_import)
    dataset_filelistings, fold_split_file_dicts = {}, {}
    for i, dataset_name in enumerate(datasets_to_import):
        dataset_filelistings[dataset_name], fold_split_file_dicts[dataset_name], dataset_stats = \
            import_dataset(data_config=data_config,
                           data_dir=data_dir,
                           dataset_name=dataset_name,
                           run_mode=run_mode,
                           exp_run=exp_run,
                           config=config)
        exp_run['DATA'] = {dataset_name: dataset_stats}

    # reverse fold and dataset_name in the fold_splits for easier processing afterwards
    fold_split_file_dicts = reverse_fold_and_dataset_order(fold_split_file_dicts)

    return fold_split_file_dicts, exp_run


def import_dataset(data_config: DictConfig,
                   data_dir: str,
                   dataset_name: str,
                   config: DictConfig,
                   exp_run: dict,
                   run_mode: str = 'train'):

    logger.info('Importing: {}', dataset_name)
    dataset_cfg = data_config['DATA_SOURCE'][dataset_name]

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
        logger.info('Data directory did not exist in "{}", creating it', data_dir)

    if not check_if_key_in_dict(data_config['DATA_SOURCE'], dataset_name):
        raise IOError('You wanted to use the dataset = "{}", but you had not defined that in your config!\n'
                      'You should have something defined for this in config["config"]["DATA"], '
                      'see MINIVESS definition for an example'.format(dataset_name))

    if dataset_name == 'MINIVESS':
        filelisting, fold_split_file_dicts, dataset_stats \
            = import_minivess_dataset(dataset_cfg=dataset_cfg,
                                      data_dir=data_dir,
                                      run_mode=run_mode,
                                      config=config,
                                      exp_run=exp_run,
                                      dataset_name=dataset_name,
                                      fetch_method=dataset_cfg['FETCH_METHOD'],
                                      fetch_params=dataset_cfg['FETCH_METHODS'][dataset_cfg['FETCH_METHOD']])

    else:
        raise NotImplementedError('Do not yet know how to download a dataset '
                                  'called = "{}"'.format(dataset_name))

    # Log the dataset to MLflow
    if config['config']['LOGGING']['MLFLOW']['TRACKING']:
        mlflow_log_dataset(mlflow_config=config['config']['LOGGING']['MLFLOW'],
                           dataset_cfg=data_config['DATA_SOURCE'][dataset_name],
                           filelisting=filelisting,
                           fold_split_file_dicts=fold_split_file_dicts,
                           config=config)

    return filelisting, fold_split_file_dicts, dataset_stats


def get_dir_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size
