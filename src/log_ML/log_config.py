import os
import wandb
import mlflow
from loguru import logger

from src.inference.ensemble_utils import get_ensemble_name, get_submodel_name
from src.log_ML.log_model_registry import log_ensembles_to_mlflow, log_ensembles_to_wandb
from src.log_ML.log_utils import write_config_as_yaml


def log_config_artifacts(log_name: str,
                         cv_dir_out: str,
                         output_dir: str,
                         config: dict,
                         fold_results: dict,
                         ensembled_results: dict,
                         cv_ensemble_results: dict,
                         experim_dataloaders: dict,
                         logging_services: list,
                         wandb_run: wandb.sdk.wandb_run.Run):
    """
    After Cross-validation of the all the folds, you are done with the config, and you can
    write the "final artifacts" then disk.
    """

    # Log the artifact dir
    logger.info('{} | ENSEMBLED Cross-Validation results | Artifacts directory'.format(logging_services))
    if 'WANDB' in logging_services:
        artifact_dir = wandb.Artifact(name=log_name, type='artifacts')
        artifact_dir.add_dir(local_path=cv_dir_out, name='CV-ENSEMBLE_artifacts')
        wandb_run.log_artifact(artifact_dir)
    if 'MLflow' in logging_services:
        mlflow.log_artifact(cv_dir_out)

    # Log all the models from all the folds and all the repeats to the Artifact Store
    # and these are accessible to Model Registry as well
    logger.info('{} | ENSEMBLED Cross-Validation results | Models to Model Registry'.format(logging_services))
    model_paths = log_model_ensemble_to_model_registry(fold_results=fold_results,
                                                       ensembled_results=ensembled_results,
                                                       cv_ensemble_results=cv_ensemble_results,
                                                       experim_dataloaders=experim_dataloaders,
                                                       log_name=log_name,
                                                       wandb_run=wandb_run,
                                                       logging_services=logging_services,
                                                       config=config,
                                                       test_loading=True)  # GET FROM CONFIG

    # HERE, log the config as .yaml file back to disk
    logger.info('{} | ENSEMBLED Cross-Validation results | Config as YAML'.format(logging_services))
    path_out, loaded_dict = write_config_as_yaml(config=config, dir_out=output_dir)
    if 'WANDB' in logging_services:
        artifact_cfg = wandb.Artifact(name='config', type='config')
        artifact_cfg.add_file(path_out)
        wandb_run.log_artifact(artifact_cfg)
    if 'MLflow' in logging_services:
        mlflow.log_dict(loaded_dict, artifact_file=os.path.split(path_out)[1])

    logger.info('{} | ENSEMBLED Cross-Validation results | Loguru log saved as .txt'.format(logging_services))
    path_out = config['run']['output_log_path']
    if 'WANDB' in logging_services:
        artifact_log = wandb.Artifact(name='log', type='log')
        artifact_log.add_file(path_out)
        wandb_run.log_artifact(artifact_log)
    if 'MLflow' in logging_services:
        mlflow.log_artifact(path_out)

    return model_paths


def log_model_ensemble_to_model_registry(fold_results: dict,
                                         ensembled_results: dict,
                                         cv_ensemble_results: dict,
                                         experim_dataloaders: dict,
                                         log_name: str,
                                         wandb_run: wandb.sdk.wandb_run.Run,
                                         logging_services: list,
                                         config: dict,
                                         test_loading: bool = False):

    # Collect and simplify the submodel structure of the ensemble(s)
    model_paths, ensemble_models_flat = collect_submodels_of_the_ensemble(fold_results)

    log_outputs = {}
    if 'WANDB' in logging_services:
        log_ensembles_to_wandb(ensemble_models_flat=ensemble_models_flat,
                               config=config,
                               wandb_run=wandb_run,
                               test_loading=test_loading)

    if 'MLflow' in logging_services:
        log_outputs['MLflow'] = (
            log_ensembles_to_mlflow(ensemble_models_flat=ensemble_models_flat,
                                    experim_dataloaders=experim_dataloaders,
                                    ensembled_results=ensembled_results,
                                    cv_ensemble_results=cv_ensemble_results,
                                    config=config,
                                    test_loading=test_loading))

    return {'model_paths': model_paths, 'ensemble_models_flat': ensemble_models_flat, 'log_outputs': log_outputs}


def collect_submodels_of_the_ensemble(fold_results: dict):

    # Two ways of organizing the same saved models, deprecate the other later
    model_paths = {}  # original nesting notation
    ensemble_models_flat = {}  # more intuitive maybe, grouping models under the same ensemble name
    n_submodels = 0
    n_ensembles = 0

    for fold_name in fold_results:
        model_paths[fold_name] = {}
        for archi_name in fold_results[fold_name]:
            for repeat_name in fold_results[fold_name][archi_name]:
                model_paths[fold_name][repeat_name] = {}
                best_dict = fold_results[fold_name][archi_name][repeat_name]['best_dict']
                for ds in best_dict:
                    model_paths[fold_name][repeat_name][ds] = {}
                    for tracked_metric in best_dict[ds]:

                        model_path = best_dict[ds][tracked_metric]['model']['model_path']
                        model_paths[fold_name][repeat_name][ds][tracked_metric] = model_path

                        ensemble_name = get_ensemble_name(dataset_validated=ds,
                                                          metric_to_track=tracked_metric)

                        if ensemble_name not in ensemble_models_flat:
                            ensemble_models_flat[ensemble_name] = {}
                            n_ensembles += 1

                        submodel_name = get_submodel_name(archi_name=archi_name,
                                                          fold_name=fold_name,
                                                          repeat_name=repeat_name)

                        if submodel_name not in ensemble_models_flat[ensemble_name]:
                            ensemble_models_flat[ensemble_name][submodel_name] = model_path
                            n_submodels += 1

    # Remember that you get more distinct ensembles if use more tracking metrics (e.g. track for best loss and for
    # best Hausdorff distance), and if you validate for more subsets of the data instead of just having one
    # "vanilla" validation splöit
    logger.info('Collected a total of {} models, and {} distinct ensembles to be logged to Model Registry'.
                format(n_submodels, n_ensembles))

    return model_paths, ensemble_models_flat
