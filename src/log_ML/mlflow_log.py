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
        experiment = mlflow.set_experiment(experiment_name)
        # mlflow.set_experiment_tag("tag_name", tag_value)  # if you want to add some tags
        logger.info(" experiment name: {}".format(experiment.experiment_id))
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
        logger.info('MLflow | Initializing MLflow Experiment tracking')
        if mlflow_config['s3'] is not None:
            logger.info('MLflow | Logging to a remote tracking MLflow Server (S3 for example)')
            raise NotImplementedError('S3 logging not tested atm')
        else:

            # see e.g. https://github.com/dmatrix/google-colab/blob/master/mlflow_issue_3317.ipynb
            local_server_path = config['run']['output_mlflow_dir']
            os.makedirs(local_server_path, exist_ok=True)
            mlflow.set_tracking_uri(local_server_path)

            logger.info('MLflow | Logging to a local MLflow Server: "{}"'.format(local_server_path))
            experiment, active_run = init_mlflow_run(local_server_path,
                                                     experiment_name=experiment_name,
                                                     run_name=run_name)

            logger.info('MLflow | Writing experiment hyperparameters (from config["hyperparameters_flat"])')
            for hyperparam_key in config["hyperparameters_flat"]:
                mlflow.log_param(hyperparam_key, config["hyperparameters_flat"][hyperparam_key])
                logger.debug(' {} = {}'.format(hyperparam_key, config["hyperparameters_flat"][hyperparam_key]))

            logger.info('MLflow | Placeholder to log your Dataset, '
                        'see https://mlflow.org/docs/latest/python_api/mlflow.data.html')

    else:
        logger.info('Skipping MLflow Experiment tracking')

    return {'experiment': experiment, 'active_run': active_run}


