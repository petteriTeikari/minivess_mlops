import os

import mlflow
from loguru import logger
import configparser

def authenticate_mlflow(fname_creds: str = 'mlflow_credentials.ini'):
    """
    https://mlflow.org/docs/latest/auth/index.html#using-credentials-file
    """
    pwd = os.getcwd()
    secrets_dir = os.path.join(pwd, '..', 'secrets')
    if not os.path.exists(secrets_dir):
        raise IOError('Cannot found secrets folder ({}), thus cannot authenticate to MLflow'.format(secrets_dir))
    else:
        credentials_file = os.path.join(secrets_dir, fname_creds)
        if not os.path.exists(credentials_file):
            raise IOError('Cannot found credentials file ({}), '
                          'thus cannot authenticate to MLflow'.format(credentials_file))
        else:
            # https://stackoverflow.com/a/8884638
            logger.info(" reading MLflow credentials from file '{}'".format(fname_creds))
            credentials = configparser.ConfigParser()
            credentials.read(credentials_file)

    # I guess you could have slightly nicer weay to handle sensitive information
    os.environ['MLFLOW_TRACKING_USERNAME'] = credentials['mlflow']['mlflow_tracking_username']
    os.environ['MLFLOW_TRACKING_PASSWORD'] = credentials['mlflow']['mlflow_tracking_password']


def init_mlflow_logging(config: dict,
                        mlflow_config: dict,
                        experiment_name: str = "MINIVESS_segmentation",
                        run_name: str = "UNet3D"):
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


    if mlflow_config['TRACKING']['enable']:
        logger.info('MLflow | Initializing MLflow Experiment tracking')
        if mlflow_config['server_URI'] is not None:
            logger.info('MLflow | Logging to a remote tracking MLflow Server ({})'.format(mlflow_config['server_URI']))
            tracking_uri = mlflow_config['server_URI']
            mlflow.set_tracking_uri(tracking_uri)
            authenticate_mlflow()
        else:
            # see e.g. https://github.com/dmatrix/google-colab/blob/master/mlflow_issue_3317.ipynb
            tracking_uri = config['run']['output_mlflow_dir']
            os.makedirs(tracking_uri, exist_ok=True)
            mlflow.set_tracking_uri(tracking_uri)

        logger.info('MLflow | Logging to a local MLflow Server: "{}"'.format(tracking_uri))
        try:
            experiment, active_run = init_mlflow_run(tracking_uri,
                                                     experiment_name=experiment_name,
                                                     run_name=run_name)
        except Exception as e:
            raise IOError('Failed to initialize the MLflow logging!\n'
                          ' - if you are using Dagshub MLflow, did you set the environment variables (your credentials?\n'
                          '   https://dagshub.com/docs/integration_guide/mlflow_tracking/#3-set-up-your-credentials\n'
                          '    e.g. export MLFLOW_TRACKING_USERNAME=<username>\n'
                          '         export MLFLOW_TRACKING_PASSWORD=<password/token>\n'
                          ' If authentication fix does not work, you can either:'
                          '   mlflow_config["TRACKING"]["enable"] = False\n'
                          '   mlflow_config["server_URI"]  = null\n'
                          'error = {}'.format(e))

        logger.info('MLflow | Writing experiment hyperparameters (from config["hyperparameters_flat"])')
        for hyperparam_key in config["hyperparameters_flat"]:
            # https://dagshub.com/docs/troubleshooting/
            mlflow.log_param(hyperparam_key, config["hyperparameters_flat"][hyperparam_key])
            logger.debug(' {} = {}'.format(hyperparam_key, config["hyperparameters_flat"][hyperparam_key]))

        logger.info('MLflow | Placeholder to log your Dataset, '
                    'see https://mlflow.org/docs/latest/python_api/mlflow.data.html')

    else:
        logger.info('Skipping MLflow Experiment tracking')

    mlflow_dict = {'experiment': experiment, 'active_run': active_run}

    # As we are using OmegaConf, you cannot just dump whatever stored in Dictionaries
    mlflow_dict_omegaconf = mlflow_dicts_to_omegaconf_dict(experiment, active_run)

    return mlflow_dict_omegaconf, mlflow_dict


def mlflow_dicts_to_omegaconf_dict(experiment, active_run):

    def convert_indiv_dict(object_in, prefix: str = None):
        dict_out = {}
        for property, value in vars(object_in).items():

            if property[0] == '_': # remove the _
                property = property[1:]

            if prefix is not None:
                key_out = prefix + property
            else:
                key_out = property

            # at the moment, just output the string values, as in names and paths
            if isinstance(value, str):
                dict_out[key_out] = value

        return dict_out


    experiment_out = convert_indiv_dict(object_in=experiment)
    mlflow_dict_out = {**experiment_out, **active_run.data.tags}

    return mlflow_dict_out