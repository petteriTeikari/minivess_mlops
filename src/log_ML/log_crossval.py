import json
import os

import wandb
from loguru import logger

import numpy as np

from src.log_ML.json_log import to_serializable
from src.log_ML.wandb_log import wandb_log_crossval


def log_cv_results(cv_results: dict,
                   cv_ensemble_results: dict,
                   fold_results: dict,
                   config: dict,
                   output_dir: str,
                   cv_averaged_output_dir: str,
                   cv_ensembled_output_dir: str):

    logger.info('Logging Cross-Validation-wise results to WANDB')
    os.makedirs(cv_averaged_output_dir, exist_ok=True)
    os.makedirs(cv_ensembled_output_dir, exist_ok=True)

    # Save the results to local (or S3) disk so that you could later also use the data easier
    save_log_cv_dicts_to_disk(cv_results=cv_results,
                              cv_ensemble_results=cv_ensemble_results,
                              cv_averaged_output_dir=cv_averaged_output_dir,
                              cv_ensembled_output_dir=cv_ensembled_output_dir)

    model_paths = wandb_log_crossval(cv_results=cv_results,
                                     cv_ensemble_results=cv_ensemble_results,
                                     fold_results=fold_results,
                                     cv_averaged_output_dir=cv_averaged_output_dir,
                                     cv_ensembled_output_dir=cv_ensembled_output_dir,
                                     output_dir=output_dir,
                                     config=config)

    return model_paths


def save_log_cv_dicts_to_disk(cv_results: dict,
                              cv_ensemble_results: dict,
                              cv_averaged_output_dir: str,
                              cv_ensembled_output_dir: str):

    cv_path_out = os.path.join(cv_averaged_output_dir, 'cv_averaged_results.json')
    json_object1 = json.dumps(cv_results, default=to_serializable)
    with open(cv_path_out, "w") as outfile:
        outfile.write(json_object1)
    logger.info('Wrote CV results to disk as .JSON ({})'.format(cv_path_out))

    cv_ensemble_path_out = os.path.join(cv_ensembled_output_dir, 'cv_ensemble_results.json')
    json_object2 = json.dumps(cv_ensemble_results, default=to_serializable)
    with open(cv_ensemble_path_out, "w") as outfile:
        outfile.write(json_object2)
    logger.info('Wrote CV Ensemble results to disk as .JSON ({})'.format(cv_ensemble_path_out))


