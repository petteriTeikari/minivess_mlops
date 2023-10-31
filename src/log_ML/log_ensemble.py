import mlflow
from loguru import logger
from omegaconf import DictConfig

from src.inference.ensemble_utils import split_ensemble_name
from src.log_ML.logging_main import correct_key_for_main_result
from src.utils.dict_utils import cfg_key


def log_ensemble_results(ensemble_results,
                         cfg: dict,
                         stat_key: str = 'mean',
                         service: str = 'MLflow',
                         fold_name: str = None):

    metric_cfg = cfg_key(cfg, 'hydra_cfg', 'config', 'LOGGING', 'MAIN_METRIC')

    if service is not None:
        logger.info('Logging ensemble metrics to experiment tracking service = {}'.format(service))
        for split in ensemble_results:
            for ensemble_name in ensemble_results[split]:
                metrics = ensemble_results[split][ensemble_name]['stats']['metrics']
                for metric in metrics:
                    stats_dict = metrics[metric]
                    dset, tracked_metric = split_ensemble_name(ensemble_name)
                    metric_name = '{}/ensemble_{}/{}/{}'.format(fold_name, metric, split, ensemble_name)
                    value = stats_dict[stat_key]
                    logger.info('{} | "{}": {:.3f}'.format(service, metric_name, value))

                    if service == 'MLflow':
                        mlflow.log_metric(metric_name, value)
                    else:
                        raise NotImplementedError('Unknown Experiment Tracking service = "{}"'.format(service))

                    metric_main = correct_key_for_main_result(metric_name=metric_name,
                                                              fold_name=fold_name,
                                                              tracked_metric=tracked_metric,
                                                              metric=metric,
                                                              dataset=dset,
                                                              split=split,
                                                              metric_cfg=metric_cfg)

                    if metric_main != metric_name:
                        if service == 'MLflow':
                            mlflow.log_metric(metric_main, value)
                            logger.info('{} (main) | "{}": {:.3f}'.format(service, metric_main, value))

    else:
        logger.warning('Not logging ensemble metrics as no Experiment Tracking services enabled'.format(service))
