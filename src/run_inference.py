import argparse
import os
import sys

import mlflow
from loguru import logger

src_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(src_path)[0]
sys.path.insert(
    0, project_path
)  # so that src. is imported correctly also in VSCode by default

from src.inference.bentoml_utils import (
    import_mlflow_model_to_bentoml,
    save_bentoml_model_to_model_store,
)
from src.log_ML.mlflow_log.mlflow_admin import get_reg_mlflow_model
from src.utils.general_utils import print_dict_to_logger

src_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(src_path)[0]
sys.path.insert(0, project_path)


def parse_args_to_dict():
    parser = argparse.ArgumentParser(description="Inference 2-PM stack from folder")
    parser.add_argument(
        "-c",
        "--task_config-file",
        type=str,
        required=True,
        default="inference/inference_folder",
        help="Name of your task-specific .yaml file, e.g. 'config_test'",
    )
    parser.add_argument(
        "-data",
        "--data_dir",
        type=str,
        required=False,
        default="/home/petteri/test_volumes",
        help="Folder of input files",
    )
    parser.add_argument(
        "-output",
        "--output_dir",
        type=str,
        required=False,
        default="/home/petteri/test_volumes_inference",
        help="Where the data is downloaded, or what dir needs to be mounted when you run this"
        "on Docker",
    )
    parser.add_argument(
        "-s3",
        "--s3_mount",
        action="store_true",
        default=None,
        required=False,
        help="None for local MLflow server, otherwise use Dagshub or some other remote uri",
    )
    parser.add_argument(
        "-uri",
        "--mlflow_server_uri",
        type=str,
        required=False,
        default=None,
        help="with None using local MLflow, otherwise specify e.g. a Dagshub uri",
    )
    parser.add_argument(
        "-p",
        "--project_name",
        type=str,
        required=False,
        default="minivess-test2",
        help="Name of the project in WANDB/MLOps. Keep the same name for all the segmentation"
        "experiments so that you can compare how tweaks affect segmentation performance."
        "Obviously create a new project if you have MINIVESS_v2 or some other dataset, when"
        "you cannot meaningfully compare e.g. DICE score from dataset 1 to dataset 2",
    )
    args_dict = vars(parser.parse_args())
    logger.info("Parsed input arguments:")
    print_dict_to_logger(args_dict, prefix="")

    return args_dict


def get_model_from_mlflow_as_bentoml_model(
    model_name: str, inference_cfg: str, server_uri: str
):
    # Get the model to be used for inference from MLflow Model Registry
    cfg, mlflow_model = get_reg_mlflow_model(
        model_name=model_name, inference_cfg=inference_cfg, server_uri=server_uri
    )

    # Import the model to BentoML deployment
    run = mlflow.get_run(mlflow_model["latest_version"].run_id)
    bento_model, pyfunc_model = import_mlflow_model_to_bentoml(
        run=run, model_uri=mlflow_model["model_uri"]
    )
    # Save to BentoML Model Store?
    save_bentoml_model_to_model_store(bento_model, pyfunc_model, mlflow_model, cfg)

    model_dict = {
        "mlflow_model": mlflow_model,
        "bento_model": bento_model,
        "pyfunc_model": pyfunc_model,
    }

    return cfg, model_dict


# e.g. with "-c inference/inference_folder" arguments
if __name__ == "__main__":
    # Get the cfg used for training and the MLflow as BentoML model
    args = parse_args_to_dict()
    cfg, model_dict = get_model_from_mlflow_as_bentoml_model(
        model_name=args["project_name"],
        inference_cfg=args["task_config_file"],
        server_uri=args["mlflow_server_uri"],
    )

    # Update the data_dir
    cfg["hydra_cfg"]["config"]["DATA"]["DATA_SOURCE"]["FOLDER"]["DATA_DIR"] = args[
        "data_dir"
    ]

    # TODO! Inference
    # Define the used data with the same function as in training
    # fold_split_file_dicts, experim_datasets, experim_dataloaders, cfg['run'] = define_experiment_data(cfg=cfg)
    # dataloader = remove_unnecessary_nesting(experim_dataloaders)
    #
    # for i, batch_data in enumerate(dataloader):
    #     test_volume = batch_data['image'][0,:,0:16,0:16,0:8].unsqueeze(0)
    #     write_tensor_as_json(tensor_in=test_volume,
    #                          filename=batch_data['metadata']['filename'][0])
