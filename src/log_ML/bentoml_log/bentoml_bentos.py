import os
import subprocess
from loguru import logger
import bentoml
from omegaconf import DictConfig

from src.log_ML.bentoml_log.bentoml_utils import parse_tag_from_exception, get_bento_CLI_build_command
from src.utils.config_utils import get_repo_dir
from src.utils.dict_utils import cfg_key


def export_bento_to_s3(model_name_in: str,
                       model_name_out: str,
                       s3_bucket: str):

    # https://docs.bentoml.org/en/latest/concepts/bento.html#bento-management-apis
    logger.info('Export Bento to S3 bucket = {}'.format(s3_bucket))
    s3_path = bentoml.export_bento(model_name_in, f'{s3_bucket}/{model_name_out}')
    logger.info('Logged Bento to path = {}'.format(s3_path))


def import_bento_from_s3(model_name: str = 'minivess-segmentor',
                         model_tag: str = 'latest',
                         s3_bucket: str = 's3://minivess-bentoml_log-models'):

    import_path = f'{s3_bucket}/{model_name}:{model_tag}.bento'
    logger.info('Importing Bento from S3 bucket = {}'.format(import_path))
    try:
        bento_import = bentoml.import_bento(import_path)
    except Exception as e:
        # e.g. bentoml_log.exceptions.BentoMLException:
        # Item 'minivess-segmentor:7aymxtt5skn7qs3t' already exists in the store <osfs '/home/petteri/bentoml_log/bentos'>
        tag = parse_tag_from_exception(str(e))
        logger.warning('Failed to import Bento from S3, e = {}'.format(e))
        try:
            # try to load locally then
            local_path = tag # f'{model_name}:{model_tag}'
            bento_import = bentoml.get(local_path)
            logger.info('Imported Bento from local path {}'.format(local_path))
        except Exception as e:
            logger.error('Failed to read the Bento even locally from {}, e = {}'.format(local_path, e))
            raise IOError('Failed to read the Bento even locally from {}, e = {}'.format(local_path, e))

    logger.debug('Bento, tag = {}'.format(bento_import.tag))
    logger.debug('Bento, creation time = {}'.format(bento_import.creation_time))

    return bento_import


def get_latest_build_tag():
    # quick'n'hacky way to get the tag
    bentos = bentoml.list()
    latest_bento = bentos[-1]
    latest_bento_tag = latest_bento.tag
    logger.info('Latest Bento tag = {} (created {})'.format(latest_bento_tag, latest_bento.creation_time))
    return latest_bento, latest_bento_tag


def log_bento_built_stdout(stdout: str):
    split_lines = stdout.split('\n')
    line_found = False
    for line in split_lines:
        if 'Successfully built Bento' in line:
            line_found = True
            logger.info(line)
    if not line_found:
        logger.warning('Bento build output maybe changed as could not find the line "Successfully built Bento"')


def bentoml_save_bento(bento_svc_cfg: DictConfig,
                       export_to_s3: bool = True,
                       run_id: str = None):

    # set working directory to the root of the repo
    cwd = os.getcwd()
    repo_dir = get_repo_dir()
    os.chdir(repo_dir)

    # build Bento locally
    cmd = get_bento_CLI_build_command()
    out = subprocess.run(cmd, capture_output=True, shell=True, text=True)
    log_bento_built_stdout(stdout=out.stdout)
    bento_build, bento_build_tag = get_latest_build_tag()

    if export_to_s3:
        export_bento_to_s3(model_name_in=cfg_key(bento_svc_cfg, 's3_model_name'),
                           model_name_out=cfg_key(bento_svc_cfg, 's3_model_name'),
                           s3_bucket=cfg_key(bento_svc_cfg, 's3_bucket'))

    # Use original working directory
    os.chdir(cwd)

    return {'bento': bento_build, 'tag': bento_build_tag}

