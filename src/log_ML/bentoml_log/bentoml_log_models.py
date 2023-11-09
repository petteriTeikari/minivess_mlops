import os
import subprocess

import bentoml
from bentoml._internal.models.model import ModelInfo
import mlflow
from omegaconf import DictConfig

from src.inference.bentoml_utils import import_mlflow_model_to_bentoml
from loguru import logger

from src.log_ML.bentoml_log.bentoml_utils import parse_tag_from_exception
from src.utils.dict_utils import cfg_key


def export_bento_model_to_s3(model_name_in: str, model_name_out: str, s3_bucket: str):
    # https://docs.bentoml.org/en/latest/concepts/bento.html#bento-management-apis
    logger.info("Export Bento Model to S3 bucket = {}".format(s3_bucket))
    s3_path = bentoml.models.export_model(
        model_name_in, f"{s3_bucket}/{model_name_out}"
    )
    logger.info("Logged to path = {}".format(s3_path))


def import_bento_model_from_s3(
    model_name: str, s3_bucket: str, model_tag: str = "latest"
):
    import_path = f"{s3_bucket}/{model_name}:{model_tag}.bentomodel"
    logger.info("Importing Bento Model from S3 bucket = {}".format(import_path))
    try:
        bento_model_import = bentoml.models.import_model(import_path)
    except Exception as e:
        # e.g.bentoml.exceptions.BentoMLException: Item 'minivess-segmentor:lhydoed546n7qs3t'
        # already exists in the store <osfs '/home/petteri/bentoml/models'>
        tag = parse_tag_from_exception(str(e))
        logger.warning(
            "Failed to import Bento from S3 (tag = {}), e = {}".format(tag, e)
        )
        try:
            # try to load locally then
            local_path = f"{model_name}:{model_tag}"
            bento_model_import = bentoml.models.get(local_path)
            logger.info("Imported Bento Model from local path {}".format(local_path))
        except Exception as e:
            logger.error(
                "Failed to read the Bento even locally from {}, e = {}".format(
                    local_path, e
                )
            )
            raise IOError(
                "Failed to read the Bento even locally from {}, e = {}".format(
                    local_path, e
                )
            )

    logger.debug("Bento Model, tag = {}".format(bento_model_import.tag))
    logger.debug(
        "Bento Model, creation time = {}".format(bento_model_import.creation_time)
    )

    return bento_model_import


def get_mlflow_run_id_of_bento(bento_model):
    # Note! This is not by default in the bento_model, we decided to save the extra fields to the "metadata"
    # when initially exporting the bentoml_log model to disk, see "import_mlflow_model_to_bentoml()"
    info: ModelInfo = bento_model.info
    try:
        metadata: dict = info.metadata
        if "run_id" in metadata:
            run_id = metadata["run_id"]
            logger.info("MLFlow run id of the BentoML model = {}".format(run_id))
        else:
            logger.info("Bento Model was not saved with the MLFlow run id!")
            run_id = None

    except Exception as e:
        logger.debug("No metadata field in ModelInfo")
        run_id = None

    return run_id


def bentoml_save_mlflow_model_to_model_store(
    run: mlflow.entities.Run,
    ensemble_name: str = "dice-MINIVESS",
    model_uri: str = "models:/minivess-test2/1",
    bento_svc_cfg: DictConfig = None,
    bento_model_name: str = "minivess-segmentor",
    s3_export: bool = True,
):
    # see, "bentoml_log list" for the saved model (as you would see your Docker images)
    bento_model, pyfunc_model = import_mlflow_model_to_bentoml(
        run=run, model_uri=model_uri, bento_model_name=bento_model_name
    )

    logger.info("BentoML Model | tag = {}".format(bento_model.tag))
    logger.info("BentoML Model | path = {}".format(bento_model.path))
    model_yaml_path = os.path.join(bento_model.path, "model.yaml")
    if os.path.exists(model_yaml_path):
        logger.info("BentoML Model | model.yaml path = {}".format(model_yaml_path))
    logger.info('See output of "bentoml models list"')
    p = subprocess.run(["bentoml", "models", "list"], capture_output=True, text=True)
    logger.info(p.stdout)
    logger.info(
        f'You can use the tag "latest" with the model: {bento_model_name}:latest'
    )

    if s3_export:
        export_bento_model_to_s3(
            model_name_in=cfg_key(bento_svc_cfg, "s3_model_name"),
            model_name_out=cfg_key(bento_svc_cfg, "s3_model_name"),
            s3_bucket=cfg_key(bento_svc_cfg, "s3_bucket"),
        )

    return bento_model, pyfunc_model
