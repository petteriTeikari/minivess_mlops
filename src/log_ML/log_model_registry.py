import time
from copy import deepcopy

from loguru import logger
import mlflow
import torch
import wandb

from src.inference.ensemble_model import ModelEnsemble, inference_ensemble_with_dataloader
from src.log_ML.mlflow_log import define_mlflow_model_uri, define_artifact_name
from src.log_ML.mlflow_tests import test_mlflow_model_registry_load


def log_ensembles_to_MLflow(ensemble_models_flat: dict,
                            experim_dataloaders: dict,
                            ensembled_results: dict,
                            cv_ensemble_results: dict,
                            config: dict,
                            test_loading: bool):
    """
    See these for example for logging ensemble of models to Model Registry in MLflow:
        https://www.databricks.com/blog/2021/09/21/managing-model-ensembles-with-mlflow.html
        https://medium.com/@pennyqxr/how-to-train-and-track-ensemble-models-with-mlflow-a1d2695e784b
        https://python.plainenglish.io/how-to-create-meta-model-using-mlflow-166aeb8666a8
    """
    model_uri = define_mlflow_model_uri()
    logger.info('MLflow | Model Registry model_uri = "{}"'.format(model_uri))

    mlflow_model_log = {}
    for i, ensemble_name in enumerate(ensemble_models_flat):
        logger.info('Ensemble #{}/{} | ensemble_name = {}'.
                    format(i + 1, len(ensemble_models_flat), ensemble_name))

        no_submodels_per_ensemble = len(ensemble_models_flat[ensemble_name])
        mlflow_model_log[ensemble_name] = {}

        for j, submodel_name in enumerate(ensemble_models_flat[ensemble_name]):
            mlflow_model_log[ensemble_name][submodel_name] = {}

            model_path = ensemble_models_flat[ensemble_name][submodel_name]
            logger.info('Submodel #{}/{} | local_path = {}'.format(j+1, no_submodels_per_ensemble, model_path))

            # Load the model
            model_dict = torch.load(model_path)
            model = deepcopy(model_dict['model'])
            best_dict = model_dict['best_dict']

            # Log the model (and register it to Model Registry)
            mlflow_model_log[ensemble_name][submodel_name] = (
                mlflow_model_logging(model=model,
                                     best_dict=best_dict,
                                     model_uri=model_uri,
                                     mlflow_config=config['config']['LOGGING']['MLFLOW'],
                                     run_params_dict=config['run'],
                                     ensemble_name=ensemble_name,
                                     submodel_name=submodel_name))

            mlflow_model_log[ensemble_name][submodel_name]['best_dict'] = best_dict


        if test_loading:
            # Test that you can download the models from the Model Registry, and that the performance
            # is exactly the same as you obtained during the training (assuming that there is no
            # stochasticity in your dataloader, like some test-time augmentation)
            logger.info('MLflow | Test that you can download model from the '
                        'Model Registry and that they are reproducible')
            test_mlflow_model_registry_load(ensemble_submodels=ensemble_models_flat[ensemble_name],
                                            mlflow_model_log=mlflow_model_log[ensemble_name],
                                            ensembled_results=ensembled_results,
                                            cv_ensemble_results=cv_ensemble_results,
                                            experim_dataloaders=experim_dataloaders,
                                            ensemble_name=ensemble_name,
                                            test_config=config['config']['LOGGING']['MLFLOW']['TEST_LOGGING'],
                                            config=config)
        else:
            logger.warning('MLflow | Skipping the model loading back from MLflow, are you sure?\n'
                           'Meant as a reproducabiloity check so that you can test that the models are loaded OK,'
                           'and give the same performance metrics as seen during the training')


def mlflow_model_logging(model, best_dict: dict, model_uri: str,
                         mlflow_config: dict, run_params_dict: dict,
                         ensemble_name: str, submodel_name: str):

    mlflow_model_log = {}
    t0 = time.time()
    artifact_name = define_artifact_name(ensemble_name, submodel_name,
                                         hyperparam_name = run_params_dict['hyperparam_name'])
    logger.info('MLflow | Logging model file to registry: {}'.format(artifact_name))

    # Log model
    # https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.log_model
    # TODO! Add requirements.txt, etc. stuff around here (get requirements.txt from Dockerfile? as we have
    #  Poetry environment here
    mlflow_model_log['model_info'] = (
        mlflow.pytorch.log_model(pytorch_model=model,
                                 # registered_model_name = artifact_name,
                                 metadata={'artifact_name': artifact_name},
                                 artifact_path="model")) # Setuptools is replacing distutils.

    # Register model
    # https://mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry
    mlflow_model_log['reg_model'] = (
        mlflow.register_model(model_uri=model_uri,
                              name=artifact_name,
                              tags={'ensemble_name': ensemble_name, 'submodel_name': submodel_name}))

    logger.info('MLflow | Model log and and registering done in {:.3f} seconds'.format(time.time() - t0))

    return mlflow_model_log


def log_ensembles_to_WANDB(ensemble_models_flat: dict,
                           config: dict,
                           wandb_run: wandb.sdk.wandb_run.Run,
                           test_loading: bool):

    # TO-FIGURE-OUT: HOW TO DO ENSEMBLES THE MOST EFFICIENTLY FOR WANDB
    # This is "reserved name" in WANDB, and if you want to use Model Registry, keep this naming
    artifact_type = 'model'

    for i, ensemble_name in enumerate(ensemble_models_flat):
        for j, submodel_name in enumerate(ensemble_models_flat[ensemble_name]):
            artifact_name = '{}__{}'.format(ensemble_name, submodel_name)
            artifact_model = wandb.Artifact(name=artifact_name, type=artifact_type)
            model_path = ensemble_models_flat[ensemble_name][submodel_name]
            artifact_model.add_file(model_path)
            logger.info('WANDB | Model file logged to registry: {}'.format(artifact_name))
            wandb_run.log_artifact(artifact_model)
