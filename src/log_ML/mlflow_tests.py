import numpy as np
from loguru import logger
# from mlflow.entities.model_registry import ModelVersion

from src.inference.ensemble_model import inference_ensemble_with_dataloader, ModelEnsemble
from src.log_ML.mlflow_log import define_artifact_name
from src.log_ML.mlflow_utils import get_model_from_mlflow_model_registry, get_subdicts_from_mlflow_model_log
from src.log_ML.model_saving import get_weight_vectors_from_best_dicts


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


def test_mlflow_model_registry_load(ensemble_submodels: dict,
                                    mlflow_model_log: dict,
                                    ensembled_results: dict,
                                    cv_ensemble_results: dict,
                                    experim_dataloaders: dict,
                                    ensemble_name: str,
                                    test_config: dict,
                                    config: dict):

    best_dicts = get_subdicts_from_mlflow_model_log(mlflow_model_log, key_str = 'best_dict')

    if test_config['CHECK_WEIGHTS']['ensemble_weights_check_local']:
        # TODO! refactor so that local vs. mlflow split is done first, so that there
        #  is no need to fetch the models multiple times

        # check the weights from the locally saved models first,
        # to inspect whether the problem is not related to Model Registry at all
        model_paths = ensemble_submodels
        ensemble_model_local = ModelEnsemble(models_of_ensemble=model_paths,
                                             model_best_dicts=best_dicts,
                                             models_from_paths=True,
                                             validation_config=config['config']['VALIDATION'],
                                             ensemble_params=config['config']['ENSEMBLE']['PARAMS'],
                                             validation_params=config['config']['VALIDATION']['VALIDATION_PARAMS'],
                                             device=config['config']['MACHINE']['IN_USE']['device'],
                                             eval_config=config['config']['VALIDATION_BEST'],
                                             precision=config['config']['TRAINING']['PRECISION'])

        ensemble_weights = ensemble_model_local.get_model_weights()
        reference_weights = get_weight_vectors_from_best_dicts(best_dicts)
        weights_ok_local = test_compare_weights(test=ensemble_weights,
                                                reference=reference_weights,
                                                compare_name='local_weight_check')

    if test_config['CHECK_WEIGHTS']['ensemble_weights_check']:

        reg_models = get_subdicts_from_mlflow_model_log(mlflow_model_log, key_str='reg_model')
        ensemble_model = create_model_ensemble_from_mlflow_model_registry(ensemble_submodels=ensemble_submodels,
                                                                          ensemble_name=ensemble_name,
                                                                          config=config,
                                                                          reg_models=reg_models,
                                                                          best_dicts=best_dicts)

        ensemble_weights = ensemble_model.get_model_weights()
        reference_weights = get_weight_vectors_from_best_dicts(best_dicts)
        weights_ok_mlflow = test_compare_weights(test = ensemble_weights,
                                                 reference = reference_weights,
                                                 compare_name='mlflow_weight_check')

    if test_config['CHECK_OUTPUT']['ensemble_level_output']:

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
        test_compare_outputs(test=ensembled_results_test,
                             reference=ensemble_results_reference)

    else:
        logger.info('MLflow | SKIP testing logged models for inference at an ensemble level')

    # if test_config['CHECK_OUTPUT']['repeat_level']:
    #     raise NotImplementedError('You could do repeat-level test as well')

    logger.info('MLflow | Done testing logged models for inference')


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
        models_of_ensemble[submodel_name] = (
            get_model_from_mlflow_model_registry(model_uri=model_uri_models))

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


def test_compare_outputs(test: dict, reference: dict) -> bool:

    a = 'placeholder'


def test_compare_weights(test: list, reference: list,
                         compare_name: str) -> bool:

    assert len(test) == len(reference), ('Number of submodels in ensemble do not match,'
                                         '{} in test, {} in reference'.format(len(test), len(reference)))

    no_submodels = len(test)
    all_good = True
    for i, (test_submodel, ref_submodel) in enumerate(zip(test, reference)):

        assert test_submodel.shape[0] == ref_submodel.shape[0], \
            ('Length of weight vectors do not match,'
             '{} in test, {} in reference'.format(test_submodel.shape[0], ref_submodel.shape[0]))

        weights_length = test_submodel.shape[0]
        are_equal = np.equal(test_submodel, ref_submodel)
        all_equal = np.all(are_equal)

        if not all_equal:
            all_good = False
            logger.debug('Submodel #{}/{} of the ensemble could not be reproduced!'.format(i+1, no_submodels))
            logger.debug(' {}      {}'.format('test', 'ref'))
            for j, (t, r) in enumerate(zip(test_submodel, ref_submodel)):
                logger.debug(' {:.4f}     {:.4f}'.format(t, r))

    if all_good:
        logger.info('MLflow | Weight check OK ({})'.format(compare_name))
    else:
        logger.warning('MLflow | Weight check FAILED ({})'.format(compare_name))

    return all_good