import mlflow
from loguru import logger

def get_model_from_mlflow_model_registry(model_uri):
    """
    not pyfunc_model: https://mlflow.org/docs/latest/model-registry.html#fetching-an-mlflow-model-from-the-model-registry
    pyfunc_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    See this
    https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.load_model
    """
    logger.info('MLflow | Fetching Pytorch model from Model Registry: {}'.format(model_uri))
    loaded_model = mlflow.pytorch.load_model(model_uri)

    return loaded_model


def get_subdicts_from_mlflow_model_log(mlflow_model_log: dict, key_str: str):

    subdicts = {}
    for submodel_name in mlflow_model_log:
        subdicts[submodel_name] = mlflow_model_log[submodel_name][key_str]

    return subdicts


def get_mlflow_model_signature(model_in,
                               input_batch,
                               gt_batch):
    """
    https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.ModelSignature
    """

    a = 1

    return None