import os
import time
import yaml
from dvc.api import DVCFileSystem
from loguru import logger
from omegaconf import DictConfig

from src.utils.general_utils import print_dict_to_logger


def get_dvc_files_of_repo(repo_dir: str,
                          repo_url: str,
                          dataset_name_lowercase: str,
                          fetch_params: dict,
                          dataset_cfg: DictConfig,
                          local_download_duplicate: bool = True,
                          use_local_repo: bool = True,
                          skip_already_downloaded: bool = True,
                          use_fs_check: bool = False):
    """
    https://dvc.org/doc/api-reference/dvcfilesystem
    """
    if use_local_repo:
        dvc_repo_path = repo_dir
        if not os.path.exists(dvc_repo_path):
            logger.error('Cannot find the repo dir "{}"'.format(dvc_repo_path))
            raise IOError('Cannot find the repo dir "{}"'.format(dvc_repo_path))
        fs = DVCFileSystem(repo_dir)
    else:
        dvc_repo_path = repo_url
        fs = DVCFileSystem(repo_url, rev='main')

    try:
        cache_dir = get_dvc_cache_dir(dvc_repo_path)
        check_for_dvc_cache_existence(cache_dir)
    except Exception as e:
        logger.error('Problem getting the DVC cache dir from DVC path "{}", e = {}'.format(dvc_repo_path, e))
        raise IOError('Problem getting the DVC cache dir from DVC path "{}", e = {}'.format(dvc_repo_path, e))

    try:
        debug_dvc_files(dvc_repo_path=repo_url,
                        dataset=dataset_name_lowercase,
                        cache_dir=cache_dir)
    except Exception as e:
        logger.warning('Problem debugging the DVC files, e = {}'.format(e))

    dvc_dataset_location = f"/{fetch_params['datapath_base']}/{dataset_name_lowercase}"
    local_path = os.path.join(repo_dir, fetch_params['datapath_base'], dataset_name_lowercase)

    if not os.path.exists(local_path):
        logger.error('Cannot find the dataset from = {} (did you do dvc pull?)'.format(local_path))
        raise IOError('Cannot find the dataset from = {} (did you do dvc pull?)'.format(local_path))

    if use_fs_check:
        raise NotImplementedError('not working atm')
        # logger.info('DVC files (fs.find) from repo "{}"'.format(repo_dir))
        # dvc_files = fs.find(repo=repo_dir,
        #                     detail=False,
        #                     dvc_only=True)
        # logger.info('Found {} DVC files in repo'.format(len(dvc_files)))
        # if len(dvc_files) == 0:
        #     raise IOError('No DVC files found in the repo ("{}")'.format(repo_dir))
    else:
        # Now this assumes that you manually pulled data "dvc pull", or in Docker / Github Actions,
        # this has been done automatically
        dvc_files = None

    # download_dvc_data(dvc_dataset_location=dvc_dataset_location,
    #                   local_path=local_path,
    #                   local_download_duplicate=local_download_duplicate,
    #                   skip_already_downloaded=skip_already_downloaded,
    #                   fs=fs)

    return local_path


def download_dvc_data(dvc_dataset_location: str,
                      local_path: str,
                      fs: DVCFileSystem,
                      local_download_duplicate: bool,
                      skip_already_downloaded: bool):
    """"""
    download_dvc_data = True
    if local_download_duplicate:
        # Seems to download the files instead of making symlinks to the shared cache in /mnt/...
        if skip_already_downloaded:
            download_dvc_data = not os.path.exists(local_path)
            if not download_dvc_data:
                logger.info('Skipping DVC download, since the data is already downloaded to "{}"'.format(local_path))

        if download_dvc_data:
            logger.warning('Making a duplicate of the data now, inspect how to improve later '
                           '(i.e. using the shared cache)')
            t0 = time.time()
            fs.get(rpath=dvc_dataset_location,
                   lpath=local_path,
                   recursive=True)
            logger.info('Fetch complete in {:.3f} seconds'.format(time.time() - t0))

    else:
        raise IOError('You need to download the Minivess with DVC with some method, '
                      'when local_download_duplicate=False')


def debug_dvc_files(dvc_repo_path: str,
                    cache_dir: str,
                    dataset: str):

    repo_filelisting = os.listdir(dvc_repo_path)
    logger.info('Repo filelisting:')
    for f in repo_filelisting:
        logger.info('  {}'.format(f))

    dvc_path = os.path.join(dvc_repo_path, '.dvc')
    try:
        dvc_filelisting = os.listdir(dvc_path)
        logger.info('DVC filelisting:')
        for f in dvc_filelisting:
            logger.info('  {}'.format(f))
    except Exception as e:
        logger.warning('DVC does not seem to be initialized in the repo? no ".dvc" folder found, e = {}'.format(e))

    try:
        dvc_config = get_dvc_config(dvc_path=dvc_path)
    except Exception as e:
        logger.warning('DVC does not seem to be initialized in the repo? no ".dvc" folder found, e = {}'.format(e))
        dvc_config = None

    check_dataset_definition(data_path=os.path.join(dvc_repo_path, 'data'),
                             dataset=dataset)

    if dvc_config is not None:
        try:
            # cache_dir = dvc_config['cache']['dir']
            cache_path = os.path.join(cache_dir, 'files', 'md5')
            logger.info('DVC cache dir = "{}"'.format(cache_dir))
        except Exception as e:
            logger.warning('Cannot find DVC cache dir from config, e = {}'.format(e))
            cache_path = None

        try:
            cached_filedirs = os.listdir(cache_path)
            no_of_cached_files = len(cached_filedirs)
            logger.info('Number of cached files = {} ("{}")'.format(no_of_cached_files, cache_dir))
        except Exception as e:
            logger.warning('Cannot find cached DVC files from "{}", e = {}'.format(cache_dir, e))


def check_dataset_definition(data_path: str,
                             dataset: str):

    dataset_dvc_file_path = os.path.join(data_path, '{}.dvc'.format(dataset))
    if not os.path.exists(dataset_dvc_file_path):
        raise IOError('Cannot find the dataset definition file "{}" for dataset = "{}"'.
                      format(dataset_dvc_file_path, dataset))
    else:
        with open(dataset_dvc_file_path, 'r') as file:
            dvc_yaml = yaml.safe_load(file)

        logger.info(f'{dataset_dvc_file_path}:')
        for outs_key in dvc_yaml:
            for i, list_item in enumerate(dvc_yaml[outs_key]):
                logger.info(' {} | list #{}'.format(outs_key, i))
                for param in list_item:
                    logger.info('  {} = {}'.format(param, list_item[param]))

        # if your "dvc pull" worked ok, you should have the files (well links to actual files
        # in the shared cache here) here in the data_path
        dataset_dir = os.path.join(data_path, dataset)
        if not os.path.exists(dataset_dir):
            raise IOError('Cannot find the dataset dir "{}"\n'
                          ' It might be that the "dvc pull" failed or you had not done that?\n'
                          ' This folder is empty when pulling the git repo, and gets populated with linked files'
                          'with the "dvc pull"'.format(dataset_dir))
        else:
            logger.info('Dataset dir "{}"'.format(dataset_dir))
            dataset_dir_filelisting = os.listdir(dataset_dir)
            logger.info('Dataset dir filelisting:')
            for f in dataset_dir_filelisting:
                logger.info('  {}'.format(f))

            if dataset == 'minivess':
                if 'raw' in dataset_dir_filelisting:
                    raw_dir = os.path.join(dataset_dir, 'raw')
                    raw_dir_filelisting = os.listdir(raw_dir)
                    logger.info('Number of "Raw" files = {} '
                                '(the volumetric 2-PM .nii.gz cubes)'.format(len(raw_dir_filelisting)))


def get_dvc_cache_dir(dvc_repo_path: str):

    dvc_path = os.path.join(dvc_repo_path, '.dvc')
    dvc_config = get_dvc_config(dvc_path=dvc_path)
    if dvc_config is not None:
        cache_dir = dvc_config['cache']['dir']
    else:
        logger.warning('Failed to get the config from "{}", no cache_dir returned")'.
                       format(dvc_path))
        cache_dir = None

    try:
        import google.colab
        # NOTE! Hard-coded now in JUpyter notebook
        cache_dir = 'volumes/minivess-dvc-cache'
        logger.warning('Running in Colab, quick fix for DVC cache dir = {}'.format(cache_dir))
    except:
        logger.debug('Not running this on Google Colab')

    return cache_dir


def check_for_dvc_cache_existence(cache_dir: str):

    try:
        if not os.path.exists(cache_dir):
            raise IOError('DVC cache dir does not exist in "{}"'.format(cache_dir))
        else:
            logger.info('DVC cache dir exists in "{}"'.format(cache_dir))
    except Exception as e:
        logger.warning('Problem checking for cache dir ({}), e = {}'.format(cache_dir, e))
        raise IOError('Problem checking for cache dir ({}), e = {}'.format(cache_dir, e))


def get_dvc_config(dvc_path: str,
                   config_fname: dict = 'config'):

    import configparser
    from typing import Dict

    def to_dict(config: configparser.ConfigParser) -> Dict[str, Dict[str, str]]:
        # https://stackoverflow.com/a/62166767
        return {section_name: dict(config[section_name]) for section_name in config.sections()}

    if not os.path.exists(dvc_path):
        logger.error('Cannot find DVC path in the repo "{}", '
                     'have you copied files to Docker correctly?'.format(dvc_path))
        try:
            logger.debug('Trying to list the files in base path of the repo')
            base_path = os.path.join(dvc_path, '..')
            repo_filelisting = os.listdir(base_path)
            logger.info('Repo filelisting:')
            for f in repo_filelisting:
                logger.info('  {}'.format(f))
        except Exception as e:
            logger.error('Failed to list the files in base path of the repo, e = {}'.format(e))
        raise IOError('Cannot find DVC path in the repo "{}")'.format(dvc_path))

    config_path = os.path.join(dvc_path, config_fname)
    if os.path.exists(config_path):
        dvc_config = configparser.ConfigParser()
        dvc_config.read(config_path)
        config_dict = to_dict(dvc_config)
        logger.info('DVC CONFIG:')
        print_dict_to_logger(config_dict)
    else:
        logger.error('Cannot find DVC config file from "{}"'.format(config_path))
        config_dict = None

    return config_dict
