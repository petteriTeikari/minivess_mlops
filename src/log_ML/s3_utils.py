import os
from loguru import logger
from omegaconf import DictConfig


def sync_artifacts_to_s3(experiment_artifacts_dir: str,
                         bucket_name: str = 'minivess-artifacts',
                         cfg: dict = None):

    s3_path = os.path.join('s3://', bucket_name)
    s3_experiments_path = os.path.join(s3_path, 'experiments')

    logger.warning('PLACEHOLDER! your chance to AWS sync the artifacts to the\n'
                   'bucket = "{}"\nfrom local_path  = {}'.format(s3_experiments_path, experiment_artifacts_dir))
