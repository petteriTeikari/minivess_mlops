import argparse
import os
import sys
from loguru import logger

from src.datasets.inference_data import remove_unnecessary_nesting
from src.inference.inference_utils import inference_per_batch
from src.log_ML.mlflow_admin import get_mlflow_model_for_inference
from src.training.experiment import define_experiment_data
from src.utils.general_utils import print_dict_to_logger

src_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(src_path)[0]
sys.path.insert(0, project_path)


def parse_args_to_dict():
    parser = argparse.ArgumentParser(description='Inference 2-PM stack from folder')
    parser.add_argument('-c', '--task_config-file', type=str, required=True,
                        default='inference/inference_folder',
                        help="Name of your task-specific .yaml file, e.g. 'config_test'")
    parser.add_argument('-data', '--data_dir', type=str, required=False,
                        default='/home/petteri/test_volumes',
                        help="Folder of input files")
    parser.add_argument('-output', '--output_dir', type=str, required=False,
                        default='/home/petteri/test_volumes_inference',
                        help="Where the data is downloaded, or what dir needs to be mounted when you run this"
                             "on Docker")
    parser.add_argument('-p', '--project_name', type=str, required=False,
                        default='minivess-test2',
                        help="Name of the project in WANDB/MLOps. Keep the same name for all the segmentation"
                             "experiments so that you can compare how tweaks affect segmentation performance."
                             "Obviously create a new project if you have MINIVESS_v2 or some other dataset, when"
                             "you cannot meaningfully compare e.g. DICE score from dataset 1 to dataset 2")
    args_dict = vars(parser.parse_args())
    logger.info('Parsed input arguments:')
    print_dict_to_logger(args_dict, prefix='')

    return args_dict


if __name__ == '__main__':

    args = parse_args_to_dict()

    # Get the model to be used for inference from MLflow Model Registry
    cfg = get_mlflow_model_for_inference(project_name=args['project_name'],
                                         inference_cfg=args['task_config_file'],
                                         data_dir=args['data_dir'],
                                         output_dir=args['output_dir'])

    # Define the used data with the same function as in training
    fold_split_file_dicts, experim_datasets, experim_dataloaders, cfg['run'] = define_experiment_data(cfg=cfg)
    dataloader = remove_unnecessary_nesting(experim_dataloaders)

    for i, batch_data in enumerate(dataloader):
        filenames = batch_data['metadata']['filename']
        logger.info('Inference on batch #{}/{} (no files = {}): {}'.
                    format(i + 1, len(dataloader), len(filenames), filenames))

        # https://mlflow.org/docs/latest/models.html#model-evaluation
        inference_per_batch(image_tensor=batch_data['image'],
                            filenames=filenames,
                            batch_data=batch_data,
                            model=cfg['mlflow_run']['loaded_ensemble'],
                            cfg=cfg)
