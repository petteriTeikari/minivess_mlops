import json
import os
from loguru import logger

from src.log_ML.json_log import to_serializable
from src.log_ML.log_utils import get_used_services
from src.log_ML.wandb_log import log_crossval_res


def log_cv_results(cv_results: dict,
                   cv_ensemble_results: dict,
                   ensembled_results: dict,
                   fold_results: dict,
                   experim_dataloaders: dict,
                   config: dict,
                   output_dir: str,
                   cv_averaged_output_dir: str,
                   cv_ensembled_output_dir: str):

    logger.info('Logging Cross-Validation-wise results')
    logging_services = get_used_services(logging_cfg=config['config']['LOGGING'])
    os.makedirs(cv_averaged_output_dir, exist_ok=True)
    os.makedirs(cv_ensembled_output_dir, exist_ok=True)

    # Save the results to local (or S3) disk so that you could later also use the data easier
    save_log_cv_dicts_to_disk(cv_results=cv_results,
                              cv_ensemble_results=cv_ensemble_results,
                              cv_averaged_output_dir=cv_averaged_output_dir,
                              cv_ensembled_output_dir=cv_ensembled_output_dir)

    model_paths = log_crossval_res(cv_results=cv_results,
                                   cv_ensemble_results=cv_ensemble_results,
                                   ensembled_results=ensembled_results,
                                   fold_results=fold_results,
                                   experim_dataloaders=experim_dataloaders,
                                   cv_averaged_output_dir=cv_averaged_output_dir,
                                   cv_ensembled_output_dir=cv_ensembled_output_dir,
                                   output_dir=output_dir,
                                   logging_services=logging_services,
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


