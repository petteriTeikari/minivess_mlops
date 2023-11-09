import argparse
import os
import sys
from loguru import logger

bentoml_path = os.path.dirname(os.path.abspath(__file__))
deployment_path = os.path.split(bentoml_path)[0]
project_path = os.path.split(deployment_path)[0]
logger.debug("project_path = {}".format(project_path))
sys.path.insert(
    0, project_path
)  # so that src. is imported correctly also in VSCode by default

from src.log_ML.bentoml_log.bentoml_containarize import containarize_bento
from src.log_ML.bentoml_log.bentoml_log_models import (
    import_bento_from_s3,
    get_mlflow_run_id_of_bento,
    import_bento_model_from_s3,
)
from src.log_ML.bentoml_log.bentoml_utils import get_latest_bento_docker
from src.log_ML.mlflow_log.mlflow_admin import import_best_model_from_model_registry
from src.log_ML.mlflow_log.mlflow_init import authenticate_mlflow, init_mlflow
from src.utils.config_utils import get_service_uris, get_repo_dir
from src.utils.dict_utils import cfg_key
from src.utils.general_utils import print_dict_to_logger


def parse_args_to_dict():
    parser = argparse.ArgumentParser(description="BentoML Docker Build check")
    parser.add_argument(
        "-mlflow_log",
        "--mlflow_model_name",
        type=str,
        required=False,
        default="minivess-test2",
        help="The name that you used to log models to MLflow Model Registry",
    )
    parser.add_argument(
        "-bentoml_log",
        "--bentoml_model_name",
        type=str,
        required=False,
        default="minivess-segmentor",
        help="The name that you used to log model to BentoML Model Store",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default="is_docker_old",
        choices=["is_docker_old"],
        help="The name that you used to log model to BentoML Model Store",
    )
    args_dict = vars(parser.parse_args())
    logger.info("Parsed input arguments:")
    print_dict_to_logger(args_dict, prefix="")

    return args_dict


def initialize_mlflow(server_uri: str):
    logger.info("Initialize MLflow, server uri = {}".format(server_uri))
    env_vars_set = authenticate_mlflow(repo_dir=get_repo_dir())
    tracking_uri = init_mlflow(server_uri=server_uri, repo_dir=get_repo_dir())
    return tracking_uri, env_vars_set


def compare_mlflow_and_bento(mlflow_model, bento_model):
    # Note! We assume that MLFlow model is always logged, but BentoML not always necessarily, thus
    # BentoML in theory could lag behind and if these do not match, you could create Bento Model from the
    # MLFlow run here
    mlflow_run_id = mlflow_model["latest_version"].run_id
    if bento_model is not None:
        bento_run_id = get_mlflow_run_id_of_bento(bento_model)
        from_same_run = mlflow_run_id == bento_run_id
        logger.info(
            "MLFlow and BentoML model run_ids match, run_id = {}".format(mlflow_run_id)
        )
    else:
        logger.warning(
            "Could not import the latest Bento model, "
            "thus cannot compare if the run_id is the same as the MLflow run id"
        )
        from_same_run = None

    return from_same_run


if __name__ == "__main__":
    args = parse_args_to_dict()
    service_uris = get_service_uris()

    # get the latest registered MLFlow model
    tracking_uri, env_vars_set = initialize_mlflow(
        server_uri=cfg_key(service_uris, "MLFLOW", "server_URI")
    )
    mlflow_model = import_best_model_from_model_registry(
        model_name=args["mlflow_model_name"],
        env_vars_set=env_vars_set,
        tracking_uri=tracking_uri,
    )

    # get the latest Bento Model from the S3 bucket
    bento_model = import_bento_model_from_s3(
        model_name=args["bentoml_model_name"],
        s3_bucket=cfg_key(service_uris, "BENTOML", "s3_bucket"),
    )

    # Compare Registered MLFlow model and the Bento imported from S3
    from_same_run = compare_mlflow_and_bento(mlflow_model, bento_model)

    # Get the latest Bento (from S3), ADD THIS IF NEEDED/WANTED

    # Get the latest Bento Docker and see its run_id
    docker_built_from_best_run_id = get_latest_bento_docker(
        docker_image=cfg_key(service_uris, "BENTOML", "docker_image"),
        run_id_tag=mlflow_model["latest_version"].run_id,
    )

    # Build Bento with Containarize
    # docker_built_from_best_run_id = False
    # if not docker_built_from_best_run_id:
    #     containarize_bento(bento_model=bento_model,
    #                        docker_image=cfg_key(service_uris, 'BENTOML', 'docker_image'))

    # if args['output'] == 'is_docker_old':
    #     if docker_built_from_best_run_id:
    #         sys.exit(0)  ## echo $? from bash
    #     else:
    #         sys.exit(1)  ## echo $? from bash
