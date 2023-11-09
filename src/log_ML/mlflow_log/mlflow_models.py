import os
import mlflow
from mlflow import MlflowClient
from loguru import logger


def import_mlflow_model(mlflow_model: dict):
    model_name = mlflow_model["latest_version"].name
    model_version = mlflow_model["latest_version"].version
    model_stage = mlflow_model["latest_version"].current_stage

    try:
        # https://stackoverflow.com/a/76347084/6412152
        client = MlflowClient(mlflow.get_tracking_uri())
        download_uri = client.get_model_version_download_uri(model_name, model_version)
        logger.debug('download_uri = "{}"'.format(download_uri))
    except Exception as e:
        logger.error(
            "Fail to get the download URI for the MLflow model! e = {}".format(e)
        )
        raise IOError(
            "Fail to get the download URI for the MLflow model! e = {}".format(e)
        )

    model_uri_version = f"models:/{model_name}/{model_version}"
    model_uri_stage = f"models:/{model_name}/{model_stage}"
    model_uri_path = os.path.join(mlflow_model["artifact_base_dir"], "MLmodel")

    try:
        loaded_model = load_model_from_registry(model_uri=model_uri_version)
    except Exception as e:
        logger.error("Fail to load the MLflow model! e = {}".format(e))
        raise IOError("Fail to load the MLflow model! e = {}".format(e))

    return model_uri_version


def load_model_from_registry(model_uri: str):
    """
    https://mlflow.org/docs/latest/model-registry.html#fetching-an-mlflow-model-from-the-model-registry
    https://python.plainenglish.io/how-to-create-meta-model-using-mlflow-166aeb8666a8
    """
    logger.debug('Trying to load registered MLflow model from "{}"'.format(model_uri))
    try:
        # https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PyFuncModel.unwrap_python_model
        loaded_meta_model = mlflow.pyfunc.load_model(
            model_uri
        )  # <class 'mlflow_log.pyfunc.model.PyFuncModel'>
        unwrapped_model = (
            loaded_meta_model.unwrap_python_model()
        )  # <class 'ModelEnsemble'>

    except Exception as e:
        logger.error(
            'Loading registered model from "{}" failed, e = {}'.format(model_uri, e)
        )
        raise IOError(
            'Loading registered model from "{}" failed, e = {}'.format(model_uri, e)
        )

    logger.info("Loaded successfully pyfunc model from = {}".format(model_uri))

    return unwrapped_model
