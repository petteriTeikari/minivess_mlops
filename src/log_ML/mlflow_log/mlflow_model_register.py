import json

from loguru import logger
import mlflow
from mlflow import MlflowClient

from src.log_ML.mlflow_log.mlflow_models import load_model_from_registry


def get_log_model_history_dict_from_run(run):
    run_id = run.info.run_id
    tags = run.data.tags
    if "mlflow_log.log-model.history" in tags:
        log_model_histories_string: str = tags["mlflow_log.log-model.history"]
        log_model = json.loads(log_model_histories_string)

        if len(log_model) == 1:
            log_model_dict: dict = log_model[0]
            metamodel_name = log_model_dict["metadata"]["metamodel_name"]
        else:
            raise NotImplementedError("Check why there are more entries or none?")

    else:
        logger.debug("No 'mlflow_log.log-model.history' in tags!")
        metamodel_name = run.data.tags["metamodel_name"]

    logger.debug('metamodel_name = "{}" from run'.format(metamodel_name))
    return metamodel_name


def transition_model_stage(name: str, version: str, stage: str = "Staging"):
    # https://mlflow.org/docs/1.8.0/model-registry.html#transitioning-an-mlflow-models-stage
    logger.info(
        'Transition model "{}" (v. {}) stage to {}'.format(name, version, stage)
    )
    client = MlflowClient()
    client.transition_model_version_stage(name=name, version=version, stage=stage)


def detransition_model_stages(name: str, version: str, stage: str = "archived"):
    logger.info("Detransition old versions of the model to stage = {}".format(stage))
    client = MlflowClient()
    for old_ver in range(1, int(version)):
        logger.info('Model version = "{}" detransitioned'.format(str(old_ver)))
        client.transition_model_version_stage(
            name=name, version=str(old_ver), stage=stage
        )


def mlflow_register_model_from_run(
    run: mlflow.entities.Run,
    stage: str = "Staging",
    project_name: str = "segmentation-minivess",
):
    # https://mlflow.org/docs/1.8.0/model-registry.html#mlflow-model-registry
    client = MlflowClient()

    # https://mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry
    metamodel_name = get_log_model_history_dict_from_run(run=run)
    # metamodel_name = run.info.run_name  ## TODO! depends on name_to_use

    # Registered model names you don't necessarily want to be as cryptic as the model log name
    # which comes from the hyperparameter sweep. In the end, you might want to have the best segmentor
    # model (or in general you want these names to be a lot more human-readable)
    reg_model_name = project_name

    # https://mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry
    logger.info("Register best model with the name = {}".format(reg_model_name))
    model_uri = f"runs:/{run.info.run_id}/{metamodel_name}"
    logger.info("model_uri = {}".format(model_uri))
    reg = mlflow.register_model(model_uri=model_uri, name=reg_model_name)

    # Set model version tag
    try:
        client.set_model_version_tag(
            name=reg_model_name,
            version=reg.version,
            key="metamodel_name",
            value=metamodel_name,
        )
    except Exception as e:
        logger.error("Failed to set tags to registered model! e = {}".format(e))
        raise IOError("Failed to set tags to registered model! e = {}".format(e))

    # Set and delete aliases on models
    client.set_registered_model_alias(
        name=reg_model_name, alias="autoreregistered", version=reg.version
    )

    # Auto-stage
    transition_model_stage(name=reg_model_name, version=reg.version, stage=stage)

    # Auto-stage previous versions to None then
    # TODO!

    # Test that you can load the model
    model_uri = f"models:/{reg_model_name}/{reg.version}"
    # TODO! autogen the requirements.txt on local machine too
    # mlflow_log.pyfunc.get_model_dependencies(model_uri)
    logger.info(
        'Testing that you can actually load the registered model from "{}"'.format(
            model_uri
        )
    )
    loaded_model = load_model_from_registry(model_uri=model_uri)

    # TODO! You should try to serve the model here as well to test the load_pickle() works
    return reg, model_uri
