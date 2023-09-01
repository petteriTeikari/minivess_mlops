import json
import os

import wandb
from loguru import logger

import numpy as np

from src.log_ML.json_log import to_serializable
# from src.log_ML.wandb_log import wandb_log_main


def log_cv_results(cv_results: dict, cv_ensemble_results: dict, config: dict, output_dir: str):

    # Save the results to local (or S3) disk so that you could later also use the data easier
    save_log_cv_dicts_to_disk(cv_results=cv_results,
                              cv_ensemble_results=cv_ensemble_results,
                              output_dir=output_dir)

    # wandb_log_main(cv_results=cv_results,
    #                cv_ensemble_results=cv_ensemble_results,
    #                config=config)


def save_log_cv_dicts_to_disk(cv_results: dict, cv_ensemble_results: dict,
                              output_dir: str):

    cv_path_out = os.path.join(output_dir, 'cv_results.json')
    json_object1 = json.dumps(cv_results, default=to_serializable)
    with open(cv_path_out, "w") as outfile:
        outfile.write(json_object1)
    logger.info('Wrote CV results to disk as .JSON ({})'.format(cv_path_out))

    cv_ensemble_path_out = os.path.join(output_dir, 'cv_ensemble_results.json')
    json_object2 = json.dumps(cv_ensemble_results, default=to_serializable)
    with open(cv_ensemble_path_out, "w") as outfile:
        outfile.write(json_object2)
    logger.info('Wrote CV Ensemble results to disk as .JSON ({})'.format(cv_ensemble_path_out))


