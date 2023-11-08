import mlflow
from loguru import logger
import wandb
from omegaconf import DictConfig

from src.inference.ensemble_model import ModelEnsemble
from src.log_ML.bentoml_log.bentoml_log_models import bentoml_save_mlflow_model_to_model_store
from src.log_ML.mlflow_log.mlflow_log import define_mlflow_model_uri
from src.log_ML.mlflow_log.mlflow_log_model import mlflow_metamodel_logging
from src.log_ML.mlflow_log.mlflow_model_register import mlflow_register_model_from_run
from src.log_ML.mlflow_log.mlflow_tests import test_mlflow_models_reproduction
from src.log_ML.mlflow_log.mlflow_utils import get_mlflow_model_signature_from_dataloader_dict
from src.utils.dict_utils import cfg_key


def log_ensembles_to_mlflow(ensemble_models_flat: dict,
                            experim_dataloaders: dict,
                            ensembled_results: dict,
                            cv_ensemble_results: dict,
                            cfg: dict,
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
                              validation_config=cfg_key(cfg, 'hydra_cfg', 'config', 'VALIDATION'),
                              ensemble_params=cfg_key(cfg, 'hydra_cfg', 'config', 'ENSEMBLE', 'PARAMS'),
                              validation_params=cfg_key(cfg, 'hydra_cfg', 'config', 'VALIDATION', 'VALIDATION_PARAMS'),
                              device=cfg_key(cfg, 'run', 'MACHINE', 'device'),
                              eval_config=cfg_key(cfg, 'hydra_cfg', 'config', 'VALIDATION_BEST'),
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
                                         run_params_dict=cfg_key(cfg, 'run', 'PARAMS'),
                                         model_uri=model_uri,
                                         ensemble_name=ensemble_name,
                                         signature=signature,
                                         cfg=cfg))

            if test_loading:
                # Test that you can download the models from the Model Registry, and that the performance
                # is exactly the same as you obtained during the training (assuming that there is no
                # stochasticity in your dataloader, like some test-time augmentation)
                logger.info('MLflow | Test that you can download model from the '
                            'Model Registry and that they are reproducible')
                test_config = cfg_key(cfg, 'hydra_cfg', 'config', 'LOGGING', 'MLFLOW', 'TEST_LOGGING')
                test_results = (
                    test_mlflow_models_reproduction(ensemble_filepaths=ensemble_models_flat[ensemble_name],
                                                    ensemble_model=ensemble_models[ensemble_name],
                                                    mlflow_model_log=mlflow_model_log[ensemble_name],
                                                    ensembled_results=ensembled_results,
                                                    cv_ensemble_results=cv_ensemble_results,
                                                    experim_dataloaders=experim_dataloaders,
                                                    ensemble_name=ensemble_name,
                                                    test_config=test_config,
                                                    cfg=cfg))
            else:
                test_results = None
                logger.warning('MLflow | Skipping the model loading back from MLflow, are you sure?\n'
                               'Meant as a reproducabiloity check so that you can test that the models are loaded OK,'
                               'and give the same performance metrics as seen during the training')

        else:
            logger.warning('At the moment the Model Registry is working only for single metamodel per run\n'
                           'implement something later if you have multiple validation datasets and/or'
                           'multiple metrics. And would you only want to register the "best model" out of these?\n'
                           'The metrics for each ensemble is either way logged to mlflow_log UI')
            test_results = None

    # 'mlflow_model_log': mlflow_model_log
    return {'test_results': test_results}


def log_ensembles_to_wandb(ensemble_models_flat: dict,
                           cfg: DictConfig,
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


def register_model_from_run(run: mlflow.entities.Run,
                            cfg: DictConfig,
                            stage: str = 'Staging',
                            project_name: str = 'segmentation-minivess',
                            services: tuple = ('mlflow_log', 'bentoml_log')):


    if 'mlflow_log' in services:
        logger.info('Registering improved model to MLflow Model Registry')
        reg, model_uri = mlflow_register_model_from_run(run=run,
                                                        stage=stage,
                                                        project_name=project_name)

        # BentoML atm depends on the existing MLflow Model Registry model
        if 'bentoml_log' in services:
            logger.info('Registering improved model to BentoML Model Store')
            bento_svc_cfg = cfg_key(cfg, 'hydra_cfg', 'config', 'SERVICES', 'BENTOML')
            bento_model, pyfunc_model = (
                bentoml_save_mlflow_model_to_model_store(run=run,
                                                         ensemble_name=run.data.tags['ensemble_name'],
                                                         model_uri=model_uri,
                                                         bento_svc_cfg=bento_svc_cfg))
        else:
            logger.info('Skip BentomL model store save')

    else:
        logger.info('Model had improved, but MLflow Model Registry model registration is skipped')

