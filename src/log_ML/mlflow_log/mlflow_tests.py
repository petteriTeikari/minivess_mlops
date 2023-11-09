import os

import mlflow
import numpy as np
from loguru import logger
from omegaconf import DictConfig

from src.inference.ensemble_model import (
    inference_ensemble_with_dataloader,
    ModelEnsemble,
)
from src.log_ML.model_saving import get_weight_vectors_from_best_dicts
from src.utils.dict_utils import cfg_key


def test_inference_loaded_mlflow_model(
    ensemble_model,
    ensemble_name: str,
    experim_dataloaders: dict,
    test_config: dict,
    ensembled_results: dict = None,
):
    dataloader, ensemble_results_reference = pick_test_dataloader(
        experim_dataloaders=experim_dataloaders,
        submodel_names=list(ensemble_model.models.keys()),
        ensemble_name=ensemble_name,
        test_config=test_config,
        ensembled_results=ensembled_results,
    )

    ensemble_results = inference_ensemble_with_dataloader(
        ensemble_model, dataloader=dataloader, split=test_config["split"]
    )

    return ensemble_results, ensemble_results_reference


def pick_test_dataloader(
    experim_dataloaders: dict,
    submodel_names: list,
    ensemble_name: str,
    test_config: dict,
    ensembled_results: dict = None,
):
    # Use a sebset of the dataloader(s) to save some time:
    fold = submodel_names[0].split("_")[0]
    split = test_config["split"]
    split_subset = test_config["split_subset"]
    logger.info(
        'Pick dataloader for reproducability testing (fold = "{}", split = "{}", split_subset (dataset) = "{}"'.format(
            fold, split, split_subset
        )
    )

    dataloader_reference = experim_dataloaders[fold][split][split_subset]
    ensembled_results_reference = ensembled_results[fold][split][ensemble_name]

    return dataloader_reference, ensembled_results_reference


def test_mlflow_models_reproduction(
    ensemble_filepaths: dict,
    ensemble_model,
    mlflow_model_log: dict,
    ensembled_results: dict,
    cv_ensemble_results: dict,
    experim_dataloaders: dict,
    ensemble_name: str,
    test_config: DictConfig,
    cfg: dict,
):
    test_results = {}
    if "CHECK_LOCAL" in test_config:
        test_results["local_models"] = local_model_tests(
            test_config=test_config,
            ensemble_model=ensemble_model,
            cfg=cfg,
            model_paths=ensemble_filepaths,
        )
    else:
        test_results["local"] = None

    if "CHECK_MLFLOW_MODELS" in test_config:
        test_results["MLflow_Models"] = mlflow_models_tests(
            model_info=mlflow_model_log["log_model"],
            test_config=test_config,
            ensemble_name=ensemble_name,
            experim_dataloaders=experim_dataloaders,
            ensembled_results=ensembled_results,
        )
    else:
        test_results["MLflow_Models"] = None

    return test_results


def local_model_tests(
    ensemble_model: ModelEnsemble, test_config: dict, cfg: dict, model_paths: dict
):
    """
    As in the first check that the models that you saved to disk make sense when loaded back
    before trying to see if something funky happened in mlflow_log.log_model and load_model
    """
    results = {}
    test_name = "check_weights"
    if test_config["CHECK_LOCAL"][test_name]:
        ensemble_model_local = ModelEnsemble(
            models_of_ensemble=model_paths,
            models_from_paths=True,
            validation_config=cfg_key(cfg, "hydra_cfg", "config", "VALIDATION"),
            ensemble_params=cfg_key(cfg, "hydra_cfg", "config", "ENSEMBLE", "PARAMS"),
            validation_params=cfg_key(
                cfg, "hydra_cfg", "config", "VALIDATION", "VALIDATION_PARAMS"
            ),
            device=cfg_key(cfg, "run", "MACHINE", "device"),
            eval_config=cfg_key(cfg, "hydra_cfg", "config", "VALIDATION_BEST"),
            # TODO! need to make this adaptive based on submodel
            precision="AMP",
        )  # config['config']['TRAINING']['PRECISION'])

        ensemble_weights = ensemble_model_local.get_model_weights()
        reference_weights = get_weight_vectors_from_best_dicts(
            best_dicts=ensemble_model.model_best_dicts
        )
        results[test_name] = test_compare_weights(
            test=ensemble_weights, reference=reference_weights, compare_name=test_name
        )

    else:
        results[test_name] = None

    return results


def mlflow_models_tests(
    model_info: mlflow.models.model.ModelInfo,
    test_config: dict,
    ensemble_name: str,
    experim_dataloaders: dict,
    ensembled_results: dict,
    test_type: str = "CHECK_MLFLOW_MODELS",
):
    results = {}

    # Fetch only once the ensemble from MLflow Models
    model_uri = model_info.model_uri
    metamodel_name = os.path.split(model_uri)[1]
    ensemble_model = create_model_ensemble_from_mlflow_models(
        metamodel_name=metamodel_name, model_uri=model_uri
    )

    test_name = "check_weights"
    if test_config[test_type][test_name]:
        ensemble_weights = ensemble_model.get_model_weights()
        reference_weights = get_weight_vectors_from_best_dicts(
            best_dicts=ensemble_model.model_best_dicts
        )
        results[test_name] = test_compare_weights(
            test=ensemble_weights,
            reference=reference_weights,
            compare_name="mlflow_weight_check",
        )
    else:
        results[test_name] = None

    test_name = "ensemble_level_output"
    if test_config[test_type][test_name]:
        # Get ensembled response from the MLflow logged models
        (
            ensembled_results_test,
            ensemble_results_reference,
        ) = test_inference_loaded_mlflow_model(
            ensemble_model=ensemble_model,
            ensemble_name=ensemble_name,
            experim_dataloaders=experim_dataloaders,
            test_config=test_config,
            ensembled_results=ensembled_results,
        )

        # Compare the obtained "test ensembled_results" to the ensembled_results
        # obtained during the training. These should match
        results[test_name], metrics = test_compare_outputs(
            test=ensembled_results_test, ref=ensemble_results_reference
        )

    else:
        results[test_name] = None

    return results


def create_model_ensemble_from_mlflow_models(
    metamodel_name: str = None, model_uri: str = None, dst_path: str = None
):
    """
    https://python.plainenglish.io/how-to-create-meta-model-using-mlflow-166aeb8666a8
    """

    # def create_mlmodel_dir(dst_path: str,
    #                        subdir: str = 'artifacts'):
    #     # MLflow download fails as the "MLmodel" subdir does not exist, and MLflow does not create it?
    #     if os.path.exists(dst_path):
    #         if not os.path.exists(os.path.join(dst_path, subdir)):
    #             os.makedirs(os.path.join(dst_path, subdir))
    #         mlmodel_path = os.path.join(dst_path, subdir, 'MLmodel')
    #         if not os.path.exists(mlmodel_path):
    #             logger.info('Creating temp dir for MLflow Models download = {}'.format(mlmodel_path))
    #             os.makedirs(mlmodel_path, exist_ok=True)
    #     else:
    #         logger.error('You are trying to download MLflow EnsembleModel to nonexisting path = "{}"'.format(dst_path))
    #         raise IOError('You are trying to download MLflow EnsembleModel to nonexisting path = "{}"'.format(dst_path))
    #
    #     return mlmodel_path

    # if you get the confusing error:
    # "raise OSError(f"No such file or directory: '{local_artifact_path}'")
    # OSError: No such file or directory: '/home/petteri/test_volumes_inference/mlflow_temp/artifacts"
    # See
    #   https://stackoverflow.com/a/69196346/6412152
    #   https://github.com/mlflow/mlflow/issues/4104#issuecomment-996715519

    # https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.PyFuncModel.unwrap_python_model
    if dst_path is None:
        loaded_meta_model = mlflow.pyfunc.load_model(model_uri)
    else:
        # create_mlmodel_dir(dst_path=dst_path)
        loaded_meta_model = mlflow.pyfunc.load_model(
            model_uri=model_uri, dst_path=dst_path
        )
    # type(loaded_meta_model)  # <class 'mlflow_log.pyfunc.model.PyFuncModel'>
    unwrapped_meta_model = loaded_meta_model.unwrap_python_model()
    # type(unwrapped_model) # <class 'src.inference.ensemble_model.ModelEnsemble'>

    return unwrapped_meta_model


def test_compare_outputs(test: dict, ref: dict, stat_key: str = "mean"):
    all_good = True
    metric_checks = {}
    for i, (test_metric, ref_metric) in enumerate(
        zip(test["stats"]["metrics"], ref["stats"]["metrics"])
    ):
        assert (
            test_metric == ref_metric
        ), "Metrics should be the same both in test ({}, " "and ref ({})".format(
            test_metric, ref_metric
        )

        test_value = test["stats"]["metrics"][test_metric][stat_key]
        ref_value = ref["stats"]["metrics"][ref_metric][stat_key]

        if ref_value != test_value:
            metric_checks[test_metric] = False
            all_good = False
            logger.debug(
                'Values ("{}") of test (from MLflow Models, {:.5f}) '
                'and ref metric "{}" (obtained during training, {:.5f}) are not the same'.format(
                    stat_key, test_value, test_metric, ref_value
                )
            )
        else:
            metric_checks[test_metric] = True

    if all_good:
        logger.info("MLflow | MLflow Models output test OK)")
        return True, metric_checks
    else:
        logger.warning(
            "MLflow | MLflow Models output test failed (inspect later the stochasticity here)"
        )
        return False, metric_checks


def test_compare_weights(test: list, reference: list, compare_name: str) -> bool:
    assert len(test) == len(reference), (
        "Number of submodels in ensemble do not match,"
        "{} in test, {} in reference".format(len(test), len(reference))
    )

    no_submodels = len(test)
    all_good = True
    for i, (test_submodel, ref_submodel) in enumerate(zip(test, reference)):
        assert test_submodel.shape[0] == ref_submodel.shape[0], (
            "Length of weight vectors do not match,"
            "{} in test, {} in reference".format(
                test_submodel.shape[0], ref_submodel.shape[0]
            )
        )

        weights_length = test_submodel.shape[0]
        are_equal = np.equal(test_submodel, ref_submodel)
        all_equal = np.all(are_equal)

        if not all_equal:
            all_good = False
            logger.debug(
                "Submodel #{}/{} of the ensemble could not be reproduced!".format(
                    i + 1, no_submodels
                )
            )
            logger.debug(" {}      {}".format("test", "ref"))
            for j, (t, r) in enumerate(zip(test_submodel, ref_submodel)):
                logger.debug(" {:.4f}     {:.4f}".format(t, r))

    if all_good:
        logger.info("MLflow | Weight check OK ({})".format(compare_name))
    else:
        logger.warning("MLflow | Weight check FAILED ({})".format(compare_name))

    return all_good
