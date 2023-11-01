import os
import glob

import monai.data.dataloader
from loguru import logger
from omegaconf import DictConfig


def import_folder_dataset(dataset_cfg: DictConfig):

    # Filelisting
    files = get_files_of_folder(data_dir=dataset_cfg['DATA_DIR'])

    # Each entry as a dict in a list for Dataset creation
    data_dicts = [{'image': image_name, 'metadata': {'filename': os.path.split(image_name)[1]}} for image_name in files]

    # Put all the files to "TEST split"
    fold_split_file_dicts = {'fold0': {'TEST': data_dicts}}

    dataset_stats = None

    return files, fold_split_file_dicts, dataset_stats
    
    
def get_files_of_folder(data_dir: str,
                        ext_wildcard: str = '*.nii*'):

    if not os.path.exists(data_dir):
        raise IOError('Input data folder "{}" does not exist!'.format(data_dir))
        logger.error('Input data folder "{}" does not exist!'.format(data_dir))

    files = glob.glob(os.path.join(data_dir, ext_wildcard))
    logger.info('Found {} files from folder {}'.format(len(files), data_dir))

    return files


def remove_unnecessary_nesting(experim_dataloaders: dict) -> monai.data.dataloader.DataLoader:
    # TODO! refactor these all a nicer idea is figured out for these
    fold_keys = list(experim_dataloaders.keys())
    dataset_keys = list(experim_dataloaders[fold_keys[0]].keys())
    split_keys = list(experim_dataloaders[fold_keys[0]][dataset_keys[0]].keys())
    dataloader = experim_dataloaders[fold_keys[0]][dataset_keys[0]][split_keys[0]]
    return dataloader