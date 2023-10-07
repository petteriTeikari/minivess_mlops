import time

import mlflow
import torch
import wandb
from loguru import logger
from mlflow.entities.model_registry import ModelVersion

from src.inference.ensemble_model import ModelEnsemble, inference_ensemble_with_dataloader
from src.log_ML.mlflow_log import define_mlflow_model_uri, define_artifact_name


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
    no_ensembles = len(ensemble_models_flat)
    reg_models = {}
    best_dicts = {}

    for i, ensemble_name in enumerate(ensemble_models_flat):
        no_submodels_per_ensemble = len(ensemble_models_flat[ensemble_name])
        reg_models[ensemble_name] = {}
        best_dicts[ensemble_name] = {}

        for j, submodel_name in enumerate(ensemble_models_flat[ensemble_name]):

            model_path = ensemble_models_flat[ensemble_name][submodel_name]

            # Load the model
            model_dict = torch.load(model_path)
            best_dicts[ensemble_name][submodel_name] = model_dict['best_dict']
            model = model_dict['model']

            # Log the model (and register it to Model Registry)
            reg_models[ensemble_name][submodel_name] = (
                mlflow_model_logging(model=model,
                                     best_dict=best_dicts[ensemble_name][submodel_name],
                                     model_uri=model_uri,
                                     mlflow_config=config['config']['LOGGING']['MLFLOW'],
                                     run_params_dict=config['run'],
                                     ensemble_name=ensemble_name,
                                     submodel_name=submodel_name))


        if test_loading:
            # Test that you can download the models from the Model Registry, and that the performance
            # is exactly the same as you obtained during the training (assuming that there is no
            # stochasticity in your dataloader, like some test-time augmentation)
            logger.info('MLflow | Test that you can download model from the '
                        'Model Registry and that they are reproducible')
            test_mlflow_model_registry_load(ensemble_submodels=ensemble_models_flat[ensemble_name],
                                            reg_models=reg_models[ensemble_name],
                                            best_dicts=best_dicts[ensemble_name],
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

    t0 = time.time()
    artifact_name = define_artifact_name(ensemble_name, submodel_name,
                                         hyperparam_name = run_params_dict['hyperparam_name'])
    logger.info('MLflow | Logging model file to registry: {}'.format(artifact_name))

    # Log model
    # https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.log_model
    # TODO! Add requirements.txt, etc. stuff around here (get requirements.txt from Dockerfile? as we have
    #  Poetry environment here
    mlflow.pytorch.log_model(pytorch_model=model, artifact_path="model") # Setuptools is replacing distutils.

    # Register model
    # https://mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry
    reg_model = mlflow.register_model(model_uri=model_uri,
                                      name=artifact_name,
                                      tags={'ensemble_name': ensemble_name, 'submodel_name': submodel_name})

    logger.info('MLflow | Model log and and registering done in {:.3f} seconds'.format(time.time() - t0))

    return reg_model


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


def get_model_from_mlflow_model_registry(model_uri):
    """
    not pyfunc_model: https://mlflow.org/docs/latest/model-registry.html#fetching-an-mlflow-model-from-the-model-registry
    pyfunc_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    See this
    https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.load_model
    """
    logger.info('MLflow | Fetching Pytorch model from Model Registry: {}'.format(model_uri))
    loaded_model = mlflow.pytorch.load_model(model_uri)

    return loaded_model


def create_model_ensemble_from_mlflow_model_registry(ensemble_submodels: dict,
                                                     ensemble_name: str,
                                                     config: dict,
                                                     reg_models: dict,
                                                     best_dicts: dict):

    # Define the models of the ensemble needed for the ModelEnsemble class
    models_of_ensemble = {}
    for j, submodel_name in enumerate(ensemble_submodels):
        artifact_name = define_artifact_name(ensemble_name, submodel_name,
                                             hyperparam_name=config['run']['hyperparam_name'])
        model_uri_models = f"models:/{artifact_name}/{reg_models[submodel_name].version}"
        models_of_ensemble[submodel_name] = get_model_from_mlflow_model_registry(model_uri=model_uri_models)

    # Create the ensembleModel class with all the submodels of the ensemble
    ensemble_model = ModelEnsemble(models_of_ensemble=models_of_ensemble,
                                   model_best_dicts=best_dicts,
                                   models_from_paths=False,
                                   validation_config=config['config']['VALIDATION'],
                                   ensemble_params=config['config']['ENSEMBLE']['PARAMS'],
                                   validation_params=config['config']['VALIDATION']['VALIDATION_PARAMS'],
                                   device=config['config']['MACHINE']['IN_USE']['device'],
                                   eval_config=config['config']['VALIDATION_BEST'],
                                   precision=config['config']['TRAINING']['PRECISION'])

    return ensemble_model


def pick_test_dataloader(experim_dataloaders: dict,
                         submodel_names: list,
                         ensemble_name: str,
                         test_config: dict,
                         ensembled_results: dict = None):

    # Use a sebset of the dataloader(s) to save some time:
    fold = submodel_names[0].split('_')[0]
    split = test_config['split']
    split_subset = test_config['split_subset']
    logger.info('Pick dataloader for reproducability testing (fold = "{}", split = "{}", split_subset (dataset) = "{}"'.
                format(fold, split, split_subset))

    dataloader_reference = experim_dataloaders[fold][split][split_subset]
    ensembled_results_reference = ensembled_results[fold][split][ensemble_name]

    return dataloader_reference, ensembled_results_reference


def test_inference_loaded_mlflow_model(ensemble_model,
                                       ensemble_name: str,
                                       experim_dataloaders: dict,
                                       test_config: dict,
                                       ensembled_results: dict = None):

    dataloader, ensemble_results_reference = (
        pick_test_dataloader(experim_dataloaders=experim_dataloaders,
                             submodel_names=list(ensemble_model.models.keys()),
                             ensemble_name=ensemble_name,
                             test_config=test_config,
                             ensembled_results=ensembled_results))

    ensemble_results = inference_ensemble_with_dataloader(ensemble_model,
                                                          dataloader=dataloader,
                                                          split=test_config['split'])

    return ensemble_results, ensemble_results_reference


def test_mlflow_model_registry_load(ensemble_submodels: dict,
                                    reg_models: dict,
                                    best_dicts: dict,
                                    ensembled_results: dict,
                                    cv_ensemble_results: dict,
                                    experim_dataloaders: dict,
                                    ensemble_name: str,
                                    test_config: dict,
                                    config: dict):

    # TOADD Test the local file as well

    if test_config['ensemble_level']:
        logger.info('MLflow | Test logged models for inference at an ensemble level')
        ensemble_model = create_model_ensemble_from_mlflow_model_registry(ensemble_submodels=ensemble_submodels,
                                                                          ensemble_name=ensemble_name,
                                                                          config=config,
                                                                          reg_models=reg_models,
                                                                          best_dicts=best_dicts)

        # Get ensembled response from the MLflow logged models
        ensembled_results_test, ensemble_results_reference = (
            test_inference_loaded_mlflow_model(ensemble_model=ensemble_model,
                                               ensemble_name=ensemble_name,
                                               experim_dataloaders=experim_dataloaders,
                                               test_config=test_config,
                                               ensembled_results=ensembled_results))

        # Compare the obtained "test ensembled_results" to the ensembled_results
        # obtained during the training. These should match
        a = 'continue_here'

    else:
        logger.info('MLflow | SKIP testing logged models for inference at an ensemble level')

    # if test_config['repeat_level']:
    #     raise NotImplementedError('You could do repeat-level test as well')

    logger.info('MLflow | Done testing logged models for inference')