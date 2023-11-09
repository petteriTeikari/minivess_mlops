import time

import mlflow
import numpy as np
from loguru import logger
from mlflow.models import ModelSignature

from src.inference.ensemble_model import ModelEnsemble
from src.log_ML.mlflow_log.mlflow_log import define_metamodel_name
from src.log_ML.mlflow_log.mlflow_models import load_model_from_registry
from src.utils.dict_utils import cfg_key
from src.utils.general_utils import get_path_size


def log_model_name(
    metamodel_name: str, hyperparam_name: str, name_to_use: str = "metamodel"
):
    if name_to_use == "metamodel":
        artifact_path = metamodel_name
    elif name_to_use == "hyperparam_name":
        artifact_path = hyperparam_name
    else:
        raise IOError("Unknown name_to_use = {}".format(name_to_use))
    logger.debug("MLflow Log Model | artifact_path = {}".format(artifact_path))

    return artifact_path


def get_the_size_of_code_path(code_paths: list, alert_threshold_MB: float = 100.0):
    # If you have a bunch of stuff in your project folder and accidentally try to upload
    # too much stuff to MLflow
    sizes_kB = []
    for code_path in code_paths:
        size_kB = get_path_size(start_path=code_path)
        logger.debug("Size of code_path = {} is {:.2f} kB".format(code_path, size_kB))
        sizes_kB.append(size_kB)

    file_sum = np.sum(sizes_kB)
    logger.info(
        "Total size of code_paths to be logged with pyfunc model = {:.2f} kB".format(
            file_sum
        )
    )
    if file_sum / 1024 > alert_threshold_MB:
        logger.warning(
            "Total size ({:.1f} MB) of code_paths is larger than {} MB, "
            'are you trying to include "too much files"'.format(
                file_sum / 1024, alert_threshold_MB
            )
        )

    return file_sum


def define_code_path(run_params: dict) -> list:
    code_paths = [run_params["src_dir"]]
    file_sum_kB = get_the_size_of_code_path(code_paths=code_paths)

    return code_paths


def mlflow_metamodel_logging(
    ensemble_model,
    model_paths: dict,
    run_params_dict: dict,
    model_uri: str,
    ensemble_name: str,
    signature: ModelSignature,
    cfg: dict,
    autoregister_models: bool = False,
    immediate_load_test: bool = False,
    name_to_use: str = "metamodel",
):
    """
    https://python.plainenglish.io/how-to-create-meta-model-using-mlflow-166aeb8666a8
    """
    mlflow_model_log = {}
    t0 = time.time()
    metamodel_name = define_metamodel_name(
        ensemble_name, hyperparam_name=run_params_dict["hyperparam_name"]
    )

    # Log model
    # https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.log_model
    logger.info(
        "MLflow | Logging (pyfunc) meta model (ensemble = {}) file to Models: {}".format(
            ensemble_name, metamodel_name
        )
    )

    # https://mlflow.org/docs/latest/models.html#how-to-log-model-with-example-containing-params
    input_data = np.zeros((4, 1, 512, 512, 27)).astype(np.float32)
    params = {"return_mask": True}
    input_example = (input_data, params)

    artifact_path = log_model_name(
        metamodel_name=metamodel_name,
        hyperparam_name=run_params_dict["hyperparam_name"],
        name_to_use=name_to_use,
    )

    # Check the code path
    code_paths: list = define_code_path(run_params=cfg["run"]["PARAMS"])

    # e.g. metamodel_name = "train_placeholder_cfg_12d24fa578a123675409cccdaac14a45__dice-MINIVESS"
    # https://santiagof.medium.com/effortless-models-deployment-with-mlflow-customizing-inference-e880cd1c9bdd
    t0 = time.time()
    logger.info("MLflow | Model log started")
    mlflow_model_log["log_model"] = mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        # https://stackoverflow.com/a/70216328/6412152
        # TODO! define some ignores for this
        code_path=code_paths,
        python_model=ModelEnsemble(
            models_of_ensemble=model_paths,
            models_from_paths=True,
            validation_config=cfg_key(cfg, "hydra_cfg", "config", "VALIDATION"),
            ensemble_params=cfg_key(cfg, "hydra_cfg", "config", "ENSEMBLE", "PARAMS"),
            validation_params=cfg_key(
                cfg, "hydra_cfg", "config", "VALIDATION", "VALIDATION_PARAMS"
            ),
            device=cfg_key(cfg, "run", "MACHINE", "device"),
            eval_config=cfg_key(cfg, "hydra_cfg", "config", "VALIDATION_BEST"),
            # TODO! need to make this adaptive based on submodel
            precision="AMP",
        ),
        signature=signature,
        # input_example=input_example,
        pip_requirements=cfg["run"]["PARAMS"]["requirements-txt_path"],
        metadata={
            "ensemble_name": ensemble_name,
            "metamodel_name": metamodel_name,
            "name_to": name_to_use,
        },
    )
    logger.info(
        "MLflow | Model logging to Models done in {:.1f} seconds".format(
            time.time() - t0
        )
    )
    mlflow.set_tag("metamodel_name", metamodel_name)
    mlflow.set_tag("ensemble_name", ensemble_name)
    logger.warning("Could you want to directly save the model to BentoML as well?")

    # if autoregister_models:
    #     # https://www.databricks.com/wp-content/uploads/2020/06/blog-mlflow-model-1.png
    #     # This does not necessarily make a lot of sense, autoregister after running some hyperparam sweeps
    #     # if you have obtained a better model from those
    #     logger.info('MLflow | Registering the model to Model Registry: {}'.
    #                 format(ensemble_name, metamodel_name))
    #     model_registry_string = '(and registering to Model Registry)'
    #     mlflow_model_log['reg_model'] = (
    #         mlflow_log.register_model(model_uri=model_uri,
    #                               name=metamodel_name,
    #                               tags={'ensemble_name': ensemble_name,
    #                                     'metamodel_name': metamodel_name}))
    # else:
    #     logger.info('MLflow | SKIP Model Registering (you can do this manually in MLflow UI then')
    #     model_registry_string = ''
    #     mlflow_model_log['reg_model'] = None
    # logger.info('MLflow | Model logging to Models {} done in {:.3f} seconds'.
    #             format(model_registry_string, time.time() - t0))
    mlflow_model_log["best_dicts"] = ensemble_model.model_best_dicts

    if immediate_load_test:
        meta_model_uri = mlflow_model_log["log_model"].model_uri
        logger.info(
            "MLflow | Immediate Model load Test from Model Registry (uri = {}".format(
                meta_model_uri
            )
        )
        _ = load_model_from_registry(model_uri=meta_model_uri)

    return mlflow_model_log
