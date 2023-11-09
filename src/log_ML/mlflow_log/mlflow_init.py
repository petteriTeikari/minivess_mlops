import os
import mlflow
from loguru import logger
from omegaconf import DictConfig

from src.log_ML.mlflow_log.mlflow_utils import mlflow_dicts_to_omegaconf_dict
from src.utils.dict_utils import cfg_key
from src.utils.general_utils import import_from_dotenv, is_docker


def init_mlflow_logging(
    hydra_cfg: DictConfig,
    run_params: dict,
    mlflow_config: DictConfig,
    experiment_name: str = "MINIVESS_segmentation",
    run_name: str = "UNet3D",
    server_uri: str = None,
):
    """
    see e.g. https://github.com/Project-MONAI/tutorials/blob/main/experiment_management/spleen_segmentation_mlflow.ipynb
    https://www.mlflow.org/docs/latest/tracking.html#where-runs-are-recorded
    """
    if mlflow_config["TRACKING"]["enable"]:
        # Where to log, local or remote
        tracking_uri = init_mlflow(
            server_uri=server_uri,
            repo_dir=cfg_key(run_params, "PARAMS", "repo_dir"),
            output_mlflow_dir=cfg_key(run_params, "PARAMS", "output_mlflow_dir"),
        )

        # Init experiment and the run
        experiment, active_run = init_mlflow_run(
            tracking_uri, experiment_name=experiment_name, run_name=run_name
        )

        # Log the (hyper)parameters to the run
        init_mlflow_params(parameters=run_params["HYPERPARAMETERS_FLAT"])

        # As we are using OmegaConf, you cannot just dump whatever stored in Dictionaries
        mlflow_dict = {"experiment": experiment, "active_run": active_run}
        mlflow_dict_omegaconf = mlflow_dicts_to_omegaconf_dict(experiment, active_run)

    else:
        logger.info("Skipping MLflow Experiment tracking")
        mlflow_dict_omegaconf, mlflow_dict = None, None

    return mlflow_dict_omegaconf, mlflow_dict


def init_mlflow(server_uri: str, repo_dir: str, output_mlflow_dir: str = None):
    logger.info("MLflow | Initializing MLflow Experiment tracking")
    try:
        if server_uri is not None:
            env_vars_set = authenticate_mlflow(repo_dir=repo_dir)
            if not env_vars_set:
                tracking_uri = mlflow_local_mlflow_init(output_mlflow_dir)
                logger.info('Local MLflow Server: "{}"'.format(tracking_uri))
            else:
                logger.info("Remote tracking MLflow Server ({})".format(server_uri))
                tracking_uri = server_uri
                mlflow.set_tracking_uri(tracking_uri)
        else:
            tracking_uri = mlflow_local_mlflow_init(output_mlflow_dir)
    except Exception as e:
        logger.error("Failed to initialize the MLflow logging! e = {}".format(e))
        raise IOError("Failed to initialize the MLflow logging! e = {}".format(e))

    return tracking_uri


def mlflow_local_mlflow_init(output_mlflow_dir: str) -> str:
    # see e.g. https://github.com/dmatrix/google-colab/blob/master/mlflow_issue_3317.ipynb
    tracking_uri = output_mlflow_dir
    os.makedirs(tracking_uri, exist_ok=True)
    logger.debug(
        'MLflow | Local MLflow Server initialized at "{}"'.format(tracking_uri)
    )
    mlflow.set_tracking_uri(tracking_uri)

    return tracking_uri


def authenticate_mlflow(repo_dir: str):
    """
    https://mlflow.org/docs/latest/auth/index.html#using-credentials-file
    https://mlflow.org/docs/latest/auth/index.html#using-environment-variables
    For your own DAGSHUB MLflow server, you need to set the environment variables:
    (Click Remote -> and you see your info)
    export MLFLOW_TRACKING_USERNAME=username
    export MLFLOW_TRACKING_PASSWORD=password

    For Github Action use, these are defined in your repo secrets then
    """
    env_vars_set = True
    if is_docker():
        logger.debug("Running code in Docker (MLflow authentication)")
        # the credentials should come now e.g. from Github Secrets
    else:
        logger.debug("Running code outside Docker (MLflow authentication)")
        import_from_dotenv(repo_dir=repo_dir)

    mlflow_tracking_username = os.getenv("MLFLOW_TRACKING_USERNAME")
    if mlflow_tracking_username is None:
        logger.warning(
            "Cannot find MLFLOW_TRACKING_USERNAME environment variable, "
            "cannot log training results to cloud! Using local MLflow server!"
        )
        env_vars_set = False

    mlflow_tracking_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if mlflow_tracking_password is None:
        logger.warning(
            "Cannot find MLFLOW_TRACKING_PASSWORD environment variable, "
            "cannot log training results to cloud! Using local MLflow server!"
        )
        env_vars_set = False

    if env_vars_set:
        logger.info(
            "MLflow | Found MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD environment variables"
        )
    else:
        logger.warning(
            "Failed remote authentication\n"
            " - did you set the environment variables for Dagshub MLflow (your credentials?\n"
            "   https://dagshub.com/docs/integration_guide/mlflow_tracking/#3-set-up-your-credentials\n"
            "    e.g. export MLFLOW_TRACKING_USERNAME=<username>\n"
            "         export MLFLOW_TRACKING_PASSWORD=<password/token>\n"
            " If authentication fix does not work, you can either:"
            '   mlflow_config["TRACKING"]["enable"] = False\n'
            '   mlflow_config["server_URI"]  = null\n'
            "error = {}"
        )

    return env_vars_set


def init_mlflow_params(parameters: DictConfig):
    if parameters is not None:
        logger.info(
            'MLflow | Writing experiment hyperparameters (from cfg["run"]["HYPERPARAMETERS_FLAT"])'
        )
        for hyperparam_key in parameters:
            # https://dagshub.com/docs/troubleshooting/
            mlflow.log_param(hyperparam_key, parameters[hyperparam_key])
            logger.debug(" {} = {}".format(hyperparam_key, parameters[hyperparam_key]))
    else:
        logger.info("MLflow | Writing dummy parameter")
        mlflow.log_param("dummy_key", "dummy_value")


def init_mlflow_run(
    local_server_path: str,
    experiment_name: str = "MINIVESS_segmentation",
    run_name: str = "UNet3D",
):
    try:
        logger.debug('Set experiment name to "{}"'.format(experiment_name))
        experiment = mlflow.set_experiment(experiment_name)
        # mlflow_log.set_experiment_tag("tag_name", tag_value)  # if you want to add some tags
        logger.info(" experiment name: {}".format(experiment.experiment_id))
        logger.info(" experiment_id: {}".format(experiment.experiment_id))
        logger.info(" artifact Location: {}".format(experiment.artifact_location))
        logger.info(" tags: {}".format(experiment.tags))
        logger.info(" lifecycle_stage: {}".format(experiment.lifecycle_stage))
    except Exception as e:
        logger.error(
            "Failed to initialize the MLflow run (Set experiment name), e = {}".format(
                e
            )
        )
        raise IOError(
            "Failed to initialize the MLflow run (Set experiment name), e = {}".format(
                e
            )
        )

    try:
        logger.debug('Start MLflow run with name "{}"'.format(run_name))
        active_run = mlflow.start_run(run_name=run_name)
        logger.info(" runName: {}".format(active_run.data.tags["mlflow.runName"]))
        logger.info(" run_id: {}".format(active_run.info.run_id))
        logger.info(
            " source.type: {}".format(active_run.data.tags["mlflow.source.type"])
        )
    except Exception as e:
        logger.error(
            "Failed to initialize the MLflow run (Start run), e = {}".format(e)
        )
        raise IOError(
            "Failed to initialize the MLflow run (Set experiment name), e = {}".format(
                e
            )
        )

    return experiment, active_run
