import mlflow
import bentoml
import numpy as np
import torch

from src.inference.ensemble_model import ModelEnsemble


def import_mlflow_model_to_bentoml(mlflow_model: dict,
                                   model_name: str = 'mlflow_model'):
    """
    https://docs.bentoml.org/en/latest/integrations/mlflow.html#import-an-mlflow-model
    """

    # https://docs.bentoml.org/en/latest/integrations/mlflow.html#attach-model-params-metrics-and-tags
    run = mlflow.get_run(mlflow_model['latest_version'].run_id)

    # https://docs.bentoml.org/en/latest/integrations/mlflow.html#loading-original-model-flavor
    bento_model = bentoml.mlflow.import_model(name=model_name,
                                              model_uri=mlflow_model['model_uri'],
                                              labels=run.data.tags,
                                              metadata={
                                                  "metrics": run.data.metrics,
                                                  "params": run.data.params,
                                              }
                                              )

    # https://docs.bentoml.org/en/latest/integrations/mlflow.html#loading-pyfunc-flavor
    pyfunc_model: mlflow.pyfunc.PyFuncModel = bentoml.mlflow.load_model(model_name)
    a = pyfunc_model.predict(np.ones((1, 1, 64, 64, 64)))
    # https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PyFuncModel.unwrap_python_model
    # metamodel: ModelEnsemble = pyfunc_model.unwrap_python_model()

