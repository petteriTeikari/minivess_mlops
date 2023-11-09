import mlflow
from loguru import logger
from omegaconf import DictConfig


def mlflow_log_dataset(
    mlflow_config: dict,
    dataset_cfg: dict,
    filelisting: dict,
    fold_split_file_dicts: dict,
    cfg: DictConfig,
):
    """
    https://mlflow.org/docs/latest/python_api/mlflow.data.html
    """
    mlflow_dataset_cfg = mlflow_config["TRACKING"]["DATASET"]
    logger.info(
        "MLflow | Placeholder to log your Dataset, "
        "see https://mlflow.org/docs/latest/python_api/mlflow.data.html"
    )


def mlflow_cv_artifacts(log_name: str, local_artifacts_dir: str):
    """
    https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_artifact
    """
    logger.info(
        "MLflow | Logging the directory {} as an artifact".format(local_artifacts_dir)
    )
    mlflow.log_artifact(local_artifacts_dir)


def define_mlflow_model_uri() -> str:
    return f"runs:/{mlflow.active_run().info.run_id}/model"


def define_artifact_name(
    ensemble_name: str,
    submodel_name: str,
    hyperparam_name: str,
    simplified: bool = True,
) -> str:
    if simplified:
        return "_{}".format(submodel_name)
    else:
        return "{}__{}__{}".format(hyperparam_name, ensemble_name, submodel_name)


def define_metamodel_name(ensemble_name: str, hyperparam_name: str):
    return "{}__{}".format(hyperparam_name, ensemble_name)
