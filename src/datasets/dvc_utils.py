import os
import time

from dvc.api import DVCFileSystem

from loguru import logger

def get_dvc_commit(dataset_dir: str,
                   make_a_local_copy: bool = True):

    a = 1


def get_dvc_files_of_repo(repo_dir: str,
                          dataset_name_lowercase: str,
                          fetch_params: dict,
                          dataset_cfg: dict,
                          local_download_duplicate: bool = True,
                          use_local_repo: bool = True,
                          skip_already_downloaded: bool = True):
    """
    https://dvc.org/doc/api-reference/dvcfilesystem
    """
    if use_local_repo:
        logger.info('DVC files fromn repo "{}"'.format(repo_dir))
        fs = DVCFileSystem(repo_dir)
    else:
        logger.info('DVC files fromn repo "{}"'.format(fetch_params['repo_url']))
        fs = DVCFileSystem(fetch_params['repo_url'], rev='main')

    dvc_files = fs.find("/", detail=False, dvc_only=True)
    logger.info('Found {} DVC files in repo'.format(len(dvc_files)))

    dvc_dataset_location = f"/{fetch_params['datapath_base']}/{dataset_name_lowercase}"
    local_path = os.path.join(repo_dir, fetch_params['datapath_base'], dataset_name_lowercase)

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
            logger.info('Fetch all the files DVC path "{}" to a local path = "{}" (copy)'.
                        format(len(dvc_files), repo_dir))
            t0 = time.time()
            fs.get(rpath=dvc_dataset_location,
                   lpath=local_path,
                   recursive=True)
            logger.info('Fetch complete in {:.3f} seconds'.format(time.time() - t0))

    else:
        raise IOError('You need to download the Minivess with DVC with some method, '
                      'when local_download_duplicate=False')

    return local_path