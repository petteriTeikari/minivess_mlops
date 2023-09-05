import os

import mlflow
from loguru import logger

def init_mlflow_logging(config: dict,
                        mlflow_config: dict,
                        experiment_name: str = "MINIVESS_segmentation",
                        run_name: str = "UNet3D") -> dict:
    """
    see e.g. https://github.com/Project-MONAI/tutorials/blob/main/experiment_management/spleen_segmentation_mlflow.ipynb
    https://www.mlflow.org/docs/latest/tracking.html#where-runs-are-recorded
    :param config:
    :return:
    """

    def init_mlflow_run(local_server_path: str,
                        experiment_name: str = "MINIVESS_segmentation",
                        run_name: str = "UNet3D"):
        experiment = mlflow.set_experiment(experiment_name,)
        # mlflow.set_experiment_tag("tag_name", tag_value)  # if you want to add some tags
        logger.info(" experiment_id: {}".format(experiment.experiment_id))
        logger.info(" artifact Location: {}".format(experiment.artifact_location))
        logger.info(" tags: {}".format(experiment.tags))
        logger.info(" lifecycle_stage: {}".format(experiment.lifecycle_stage))

        active_run = mlflow.start_run(run_name=run_name)
        logger.info(" runName: {}".format(active_run.data.tags['mlflow.runName']))
        logger.info(" run_id: {}".format(active_run.info.run_id))
        logger.info(" source.type: {}".format(active_run.data.tags['mlflow.source.type']))
        return experiment, active_run


    if mlflow_config['Tracking']:
        logger.info('Initializing MLflow Experiment tracking')
        if mlflow_config['s3'] is not None:
            logger.info('Logging to a remote tracking MLflow Server (S3 for example)')
            raise NotImplementedError('S3 logging not tested atm')
        else:

            # see e.g. https://github.com/dmatrix/google-colab/blob/master/mlflow_issue_3317.ipynb
            local_server_path = os.path.join(config['ARGS']['output_dir'], 'mlflow')
            os.makedirs(local_server_path, exist_ok=True)
            mlflow.set_tracking_uri(local_server_path)

            logger.info('Logging to a local MLflow Server: "{}"'.format(local_server_path))
            experiment, active_run = init_mlflow_run(local_server_path)

            # FIXME! now you would have all the parameters that you possibly need in the "config", and you
            #  could simply dump it all? recursively write nested dict? downside is that you will have a lot
            #  of parameters that might make MLflow just messy, or do a manual LUT for a subset of params
            #  that will be the hyperparameters that you most likely are interested in changing
            logger.debug('Writing a placeholder parameter to MLflow: "{}": {}'.format('placeholder_param', 1))
            mlflow.log_param('placeholder_param', 1)

    else:
        logger.info('Skipping MLflow Experiment tracking')

    return {'experiment': experiment, 'active_run': active_run}


def mlflow_log_best_repeats(best_repeat_dicts: dict, config: dict,
                            splits: tuple = ('VAL', 'TEST')):

    logger.info('Logging (MLflow) the metrics obtained from best repeat')
    for dataset_train in best_repeat_dicts:
        for tracked_metric in best_repeat_dicts[dataset_train]:
            for split in best_repeat_dicts[dataset_train][tracked_metric]:
                if split in splits:
                    for dataset_eval in best_repeat_dicts[dataset_train][tracked_metric][split]:
                        for metric in best_repeat_dicts[dataset_train][tracked_metric][split][dataset_eval]:
                            best_repeat = best_repeat_dicts[dataset_train][tracked_metric][split][dataset_eval][metric]
                            metric_name = 'bestRepeat_{}_{}/{}/{}/{}'.format(metric, split, dataset_train,
                                                                             tracked_metric, dataset_eval)
                            metric_value = best_repeat['best_value']
                            logger.info('"{}": {:.3f}'.format(metric_name, metric_value))

                            # FIXME! basically this is the only MLflow specific call, and you could write the
                            #  metrics here to multiple platforms at once, or have a switch, up to you
                            mlflow.log_metric(metric_name, metric_value)