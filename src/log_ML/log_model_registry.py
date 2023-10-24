import os
import time
from loguru import logger
import mlflow
import wandb
from mlflow.models import ModelSignature

from src.inference.ensemble_model import ModelEnsemble
from src.log_ML.mlflow_log import define_mlflow_model_uri, define_metamodel_name
from src.log_ML.mlflow_tests import test_mlflow_models_reproduction
from src.log_ML.mlflow_utils import get_mlflow_model_signature_from_dataloader_dict


def log_ensembles_to_mlflow(ensemble_models_flat: dict,
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
    ensemble_models = {}

    for i, ensemble_name in enumerate(ensemble_models_flat):
        if i == 0:
            logger.info('Ensemble #{}/{} | ensemble_name = {}'.
                        format(i + 1, len(ensemble_models_flat), ensemble_name))

            # best_dicts = get_subdicts_from_mlflow_model_log(mlflow_model_log, key_str = 'best_dict')
            ensemble_models[ensemble_name] = (
                ModelEnsemble(models_of_ensemble=ensemble_models_flat[ensemble_name],
                              models_from_paths=True,
                              validation_config=config['config']['VALIDATION'],
                              ensemble_params=config['config']['ENSEMBLE']['PARAMS'],
                              validation_params=config['config']['VALIDATION']['VALIDATION_PARAMS'],
                              device=config['config']['MACHINE']['IN_USE']['device'],
                              eval_config=config['config']['VALIDATION_BEST'],
                              # TODO! need to make this adaptive based on submodel
                              precision='AMP'))  # config['config']['TRAINING']['PRECISION'])

            try:
                signature = get_mlflow_model_signature_from_dataloader_dict(model_in=ensemble_models[ensemble_name],
                                                                            experim_dataloaders=experim_dataloaders)
            except Exception as e:
                logger.warning('Problem creating MLflow ModelSignature, e = {}'.format(e))
                signature = None

            # Log the model to Models (and register it to Model Registry)
            mlflow_model_log[ensemble_name] = (
                mlflow_metamodel_logging(ensemble_model=ensemble_models[ensemble_name],
                                         model_paths=ensemble_models_flat[ensemble_name],
                                         run_params_dict=exp_run['RUN'],
                                         model_uri=model_uri,
                                         ensemble_name=ensemble_name,
                                         signature=signature,
                                         config=config))

            if test_loading:
                # Test that you can download the models from the Model Registry, and that the performance
                # is exactly the same as you obtained during the training (assuming that there is no
                # stochasticity in your dataloader, like some test-time augmentation)
                logger.info('MLflow | Test that you can download model from the '
                            'Model Registry and that they are reproducible')
                test_results = (
                    test_mlflow_models_reproduction(ensemble_filepaths=ensemble_models_flat[ensemble_name],
                                                    ensemble_model=ensemble_models[ensemble_name],
                                                    mlflow_model_log=mlflow_model_log[ensemble_name],
                                                    ensembled_results=ensembled_results,
                                                    cv_ensemble_results=cv_ensemble_results,
                                                    experim_dataloaders=experim_dataloaders,
                                                    ensemble_name=ensemble_name,
                                                    test_config=config['config']['LOGGING']['MLFLOW']['TEST_LOGGING'],
                                                    config=config))
            else:
                test_results = None
                logger.warning('MLflow | Skipping the model loading back from MLflow, are you sure?\n'
                               'Meant as a reproducabiloity check so that you can test that the models are loaded OK,'
                               'and give the same performance metrics as seen during the training')

        else:
            logger.warning('At the moment the Model Registry is working only for single metamodel per run\n'
                           'implement something later if you have multiple validation datasets and/or'
                           'multiple metrics. And would you only want to register the "best model" out of these?\n'
                           'The metrics for each ensemble is either way logged to mlflow UI')
            test_results = None

    # 'mlflow_model_log': mlflow_model_log
    return {'test_results': test_results}


def mlflow_metamodel_logging(ensemble_model,
                             model_paths: dict,
                             run_params_dict: dict,
                             model_uri: str,
                             ensemble_name: str,
                             signature: ModelSignature,
                             config: dict,
                             autoregister_models: bool = False,
                             immediate_load_test: bool = False):
    """
    https://python.plainenglish.io/how-to-create-meta-model-using-mlflow-166aeb8666a8
    """
    mlflow_model_log = {}
    t0 = time.time()
    metamodel_name = define_metamodel_name(ensemble_name,
                                           hyperparam_name=run_params_dict['hyperparam_name'])

    # Log model
    # https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.log_model
    logger.info('MLflow | Logging (pyfunc) meta model (ensemble = {}) file to Models: {}'.
                format(ensemble_name, metamodel_name))

    validation_params = config['config']['VALIDATION']['VALIDATION_PARAMS']
    mlflow_model_log['log_model'] = (
        mlflow.pyfunc.log_model(artifact_path=metamodel_name,
                                python_model=ModelEnsemble(models_of_ensemble=model_paths,
                                                           models_from_paths=True,
                                                           validation_config=config['config']['VALIDATION'],
                                                           ensemble_params=config['config']['ENSEMBLE']['PARAMS'],
                                                           validation_params=validation_params,
                                                           device=config['config']['MACHINE']['IN_USE']['device'],
                                                           eval_config=config['config']['VALIDATION_BEST'],
                                                           # TODO! need to make this adaptive based on submodel
                                                           precision='AMP'),
                                signature=signature,
                                pip_requirements=os.path.join(exp_run['RUN']['repo_dir'], 'requirements.txt')
                                )
    )

    if autoregister_models:
        # https://www.databricks.com/wp-content/uploads/2020/06/blog-mlflow-model-1.png
        logger.info('MLflow | Registering the model to Model Registry: {}'.
                    format(ensemble_name, metamodel_name))
        model_registry_string = '(and registering to Model Registry)'
        mlflow_model_log['reg_model'] = (
            mlflow.register_model(model_uri=model_uri,
                                  name=metamodel_name,
                                  tags={'ensemble_name': ensemble_name, 'metamodel_name': metamodel_name}))
    else:
        logger.info('MLflow | SKIP Model Registering (you can do this manually in MLflow UI then')
        model_registry_string = ''
        mlflow_model_log['reg_model'] = None

    mlflow_model_log['best_dicts'] = ensemble_model.model_best_dicts

    logger.info('MLflow | Model logging to Models {} done in {:.3f} seconds'.
                format(model_registry_string, time.time() - t0))

    if immediate_load_test:
        meta_model_uri = mlflow_model_log['log_model'].model_uri
        logger.info('MLflow | Immediate Model load Test from Model Registry (uri = {}'.format(meta_model_uri))
        try:
            # https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PyFuncModel.unwrap_python_model
            loaded_meta_model = mlflow.pyfunc.load_model(meta_model_uri)
            # type(loaded_meta_model)  # <class 'mlflow.pyfunc.model.PyFuncModel'>
            unwrapped_model = loaded_meta_model.unwrap_python_model()
            # type(unwrapped_model) # <class 'src.inference.ensemble_model.ModelEnsemble'>

        except Exception as e:
            logger.warning('Load test failed, e = {}'.format(e))

    return mlflow_model_log


def log_ensembles_to_wandb(ensemble_models_flat: dict,
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
