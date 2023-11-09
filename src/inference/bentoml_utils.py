import mlflow
import bentoml
import numpy as np
import torch
from loguru import logger
from src.inference.ensemble_model import ModelEnsemble


def test_bento_model_inference(pyfunc_model: mlflow.pyfunc.PyFuncModel):
    # https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PyFuncModel.unwrap_python_model
    # metamodel: ModelEnsemble = pyfunc_model.unwrap_python_model()
    try:
        mask_out = pyfunc_model.predict(
            np.ones((1, 1, 64, 64, 64)).astype(np.float32), {"return_mask": True}
        )
    except Exception as e:
        logger.error("Inference failed for BentoML model, e = {}".format(e))
        raise IOError("Inference failed for BentoML model, e = {}".format(e))


def import_mlflow_model_to_bentoml(
    run: mlflow.entities.Run,
    model_uri: str = "models:/minivess-test2/1",
    bento_model_name: str = "minivess-segmentor",
):
    """
    https://docs.bentoml.org/en/latest/integrations/mlflow.html#import-an-mlflow-model
    https://docs.bentoml.org/en/latest/integrations/mlflow.html#attach-model-params-metrics-and-tags
    https://docs.bentoml.org/en/latest/integrations/mlflow.html#loading-original-model-flavor
    """
    bento_model: bentoml._internal.models.model.Model = bentoml.mlflow.import_model(
        name=bento_model_name,
        model_uri=model_uri,
        labels=run.data.tags,
        metadata={
            "metrics": run.data.metrics,
            "params": run.data.params,
            "tags": run.data.tags,
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
        },
    )

    # https://docs.bentoml.org/en/latest/integrations/mlflow.html#loading-pyfunc-flavor
    pyfunc_model: mlflow.pyfunc.PyFuncModel = bentoml.mlflow.load_model(
        bento_model_name
    )
    test_bento_model_inference(pyfunc_model)

    return bento_model, pyfunc_model


def save_bentoml_model_to_model_store(
    bento_model: bentoml._internal.models.model.Model,
    pyfunc_model: mlflow.pyfunc.PyFuncModel,
    mlflow_model: dict,
    cfg: dict,
):
    # https://docs.bentoml.org/en/latest/concepts/model.html#save-a-trained-model
    logger.debug("Placeholder for model save")

    # bentoml_log.pytorch.save_model(
    #     "demo_bentoml_model_minivess",  # Model name in the local Model Store
    #     bento_model,  # Model instance being saved
    #     labels={  # User-defined labels for managing models in BentoCloud or Yatai
    #         "owner": "nlp_team",
    #         "stage": "dev",
    #     },
    #     metadata={  # User-defined additional metadata
    #         "acc": 0.1111,
    #         "cv_stats": 0.2222,
    #         "dataset_version": "20210820",
    #     },
    # )
