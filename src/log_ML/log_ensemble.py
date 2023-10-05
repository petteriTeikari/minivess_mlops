import mlflow
from loguru import logger

from src.log_ML.logging_main import correct_key_for_main_result


def log_ensemble_results(ensemble_results, config: dict,
                         stat_key: str = 'mean',
                         service: str = 'MLflow'):

    logger.info('Logging ensemble metrics to experiment tracking service = {}'.format(service))

    for split in ensemble_results:
        split_stats = ensemble_results[split]['stats']
        for dset in split_stats:
            for tracked_metric in split_stats[dset]:
                metrics = split_stats[dset][tracked_metric]['metrics']
                for metric in metrics:
                    stats_dict = metrics[metric]
                    metric_name = 'ensemble_{}_{}/{}/{}'.format(metric, split, dset, tracked_metric)
                    value = stats_dict[stat_key]
                    logger.info('{} | "{}": {:.3f}'.format(service, metric_name, value))

                    if service == 'MLflow':
                        mlflow.log_metric(metric_name, value)
                    else:
                        raise NotImplementedError('Unknown Experiment Tracking service = "{}"'.format(service))

                    metric_main = correct_key_for_main_result(metric_name=metric_name,
                                                              tracked_metric=tracked_metric, metric=metric,
                                                              dataset=dset, split=split,
                                                              metric_cfg=config['config']['LOGGING']['MAIN_METRIC'])

                    if metric_main != metric_name:
                        if service == 'MLflow':
                            mlflow.log_metric(metric_main, value)
                            logger.info('{} (main) | "{}": {:.3f}'.format(service, metric_main, value))