import json
import os
import shutil
import glob

import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from mlflow.store.entities import PagedList
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.inference.ensemble_model import ModelEnsemble
from src.log_ML.log_config import get_cfg_yaml_fname, get_run_params_yaml_fname, define_ensemble_submodels_dir_name
from src.log_ML.mlflow_init import authenticate_mlflow, init_mlflow
from src.log_ML.mlflow_tests import create_model_ensemble_from_mlflow_models
from src.utils.config_utils import get_repo_dir, get_config_dir, config_import_script
from src.utils.dict_utils import cfg_key
from src.utils.general_utils import import_from_dotenv, is_docker


def mlflow_update_best_model(project_name: str,
                             stage: str = 'Staging',
                             best_metric_name: str = 'Dice'):

    # Get best run from all the runs so far
    best_run = get_best_run(project_name,
                            best_metric_name=best_metric_name)

    # Check the best metric(s) from the registered models
    _, _, register_best_run_as_best_registered_model = (
        get_best_registered_model(project_name=project_name,
                                  best_run=best_run,
                                  best_metric_name=best_metric_name))

    # Register the model from the best run from MLflow experiments,
    # if the best run is better than the best registered model
    if register_best_run_as_best_registered_model:
        logger.info('Register the best run as the best registered model')
        register_model_from_run(run=best_run,
                                stage=stage,
                                project_name=project_name)
    else:
        logger.info('Keeping the best registered model as the best model')


def get_best_registered_model(project_name: str,
                              best_run = None,
                              best_metric_name: str = 'Dice'):

    register_best_run_as_best_registered_model = False
    if best_run is not None:
        prev_best = best_run.data.metrics[best_metric_name]

    client = MlflowClient()
    rmodels = client.search_registered_models()
    if len(rmodels) == 0:
        logger.warning('Could not find any registered models from MLflow! This run is considered the best then')
        register_best_run_as_best_registered_model = True
        rmodel, latest_ver = None, None

    else:
        for rmodel in rmodels:
            # TODO! you need to loop the prev_best to match this
            if project_name in rmodel.name:

                latest_ver = rmodel.latest_versions[0]
                run_id = latest_ver.run_id
                best_value = get_best_metric_of_run(run_id=run_id,
                                                    best_metric_name=best_metric_name)

                logger.info('registered model name "{}" '
                            '(nr of latest versions = {}, latest version = {})'.
                            format(rmodel.name, len(rmodel.latest_versions), latest_ver.version))

                if best_run is not None:
                    if best_value is None or best_value < prev_best:
                        logger.info('Best run is better than the best registered model')
                        register_best_run_as_best_registered_model = True
                    else:
                        logger.info('Best registered model does not need to be updated')
                        register_best_run_as_best_registered_model = False

    return rmodel, latest_ver, register_best_run_as_best_registered_model


def get_best_metric_of_run(run_id: str,
                           best_metric_name: str = 'Dice'):

    run = mlflow.get_run(run_id)
    if best_metric_name in run.data.metrics:
        best_value = run.data.metrics[best_metric_name]
    else:
        logger.warning('It seems that you have never registered '
                       'a model with the metric = "{}" even computed'.format(best_metric_name))
        best_value = None

    return best_value


def register_model_from_run(run, stage: str = 'Staging',
                            project_name: str = 'segmentation-minivess'):

    # https://mlflow.org/docs/1.8.0/model-registry.html#mlflow-model-registry
    client = MlflowClient()

    # https://mlflow.org/docs/latest/model-registry.html#adding-an-mlflow-model-to-the-model-registry
    log_model_dict = get_log_model_dict_from_run(run=run)
    metamodel_name = get_metamodel_name_from_log_model(log_model_dict)  # artifact_path

    # Registered model names you don't nececcsarily want to be as cryptic as the model log name
    # which comes from the hyperparameter sweep. In the end, you might want to have the best segmentor
    # model (or in general you want these names to be a lot more human-readable)
    reg_model_name = project_name

    logger.info('Register best model with the name = {}'.format(reg_model_name))
    model_uri = f"runs:/{run.info.run_id}/{reg_model_name}"
    reg = mlflow.register_model(model_uri, reg_model_name)

    # Set model version tag
    # mlflow.exceptions.RestException: RESOURCE_DOES_NOT_EXIST
    # client.set_model_version_tag(name=reg_model_name,
    #                              version=reg.version,
    #                              key="key",
    #                              value="value")

    # Set and delete aliases on models
    client.set_registered_model_alias(name=reg_model_name,
                                      alias="autoreregistered",
                                      version=reg.version)

    # Auto-stage
    transition_model_stage(name=reg_model_name,
                           version=reg.version,
                           stage=stage)

def transition_model_stage(name: str,
                           version: str,
                           stage: str = 'Staging'):
    # https://mlflow.org/docs/1.8.0/model-registry.html#transitioning-an-mlflow-models-stage
    logger.info('Transition model "{}" (v. {}) stage to {}'.format(name, version, stage))
    client = MlflowClient()
    client.transition_model_version_stage(
        name=name,
        version=version,
        stage=stage
    )


def get_log_model_dict_from_run(run):

    run_id = run.info.run_id
    tags = run.data.tags
    log_model_histories_string: str = tags['mlflow.log-model.history']
    log_model = json.loads(log_model_histories_string)

    if len(log_model) == 1:
        log_model_dict: dict = log_model[0]
    else:
        raise NotImplementedError('Check why there are more entries or none?')

    return log_model_dict


def get_mlflow_ordering_direction(best_metric_name):

    if best_metric_name == 'Dice':
        metric_best = f'{best_metric_name} DESC'
    elif best_metric_name == 'Hausdorff':
        metric_best = f'{best_metric_name} ASC'
    else:
        raise NotImplementedError('Check the metric name!, best_metric_name = {}'.format(best_metric_name))

    return metric_best


def get_best_run(project_name: str,
                 best_metric_name: str = 'Dice'):

    metric_best = get_mlflow_ordering_direction(best_metric_name)
    try:
        runs = get_runs_of_experiment(project_name=project_name,
                                      metric_best=metric_best,
                                      max_results=1)
    except Exception as e:
        logger.error('Failed to get the runs from experiment = {}! e = {}'.format(project_name, e))
        raise IOError('Failed to get the runs from experiment = {}! e = {}'.format(project_name, e))

    if len(runs) > 0:
        best_run = runs[0]
        metric_name = metric_best.split(' ')[0]
        logger.info('Best run from MLflow Tracking experiments, '
                    '{} = {:.3f}'.format(metric_name, best_run.data.metrics[metric_name]))
    else:
        logger.warning('No runs returned!')
        best_run = None

    return best_run


def get_runs_of_experiment(project_name: str,
                           max_results: int = 3,
                           metric_best: str = 'Dice DESC') -> PagedList:

    # https://mlflow.org/docs/latest/search-runs.html#python
    order_by = ["metrics.{}".format(metric_best)]
    filter_string=""
    logger.info('Get a list of the best {} runs from experiment = {}'.format(max_results, project_name))
    logger.info('Ordering of model performance based on {}'.format(order_by))
    logger.info('Filtering string {}'.format(filter_string))
    logger.info('Returning only ACTIVE runs')
    # ADD info.status == 'FINISHED'
    runs = MlflowClient().search_runs(
        experiment_ids=get_current_id(project_name=project_name),
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=max_results,
        order_by=order_by,
    )

    return runs


def get_current_id(project_name: str):
    current_experiment = dict(mlflow.get_experiment_by_name(project_name))
    return current_experiment['experiment_id']


def get_metamodel_name_from_log_model(log_model_dict: dict):
    return log_model_dict['artifact_path']


def get_mlflow_model_for_inference(project_name: str,
                                   inference_cfg: str,
                                   server_uri: str = 'https://dagshub.com/petteriTeikari/minivess_mlops.mlflow',
                                   data_dir: str = None,
                                   output_dir: str = None,
                                   model_download_method: str = 'subdirs'):

    client = MlflowClient()
    env_vars_set = authenticate_mlflow(repo_dir=get_repo_dir())

    # Where to log, local or remote
    # TODO! get server_uri from some func that knows how to import it from hierachical config
    tracking_uri = init_mlflow(server_uri=server_uri,
                               repo_dir=get_repo_dir(),
                               output_mlflow_dir=os.path.join(get_repo_dir(), 'mlflow_temp'))

    rmodel, latest_model_version, _ = get_best_registered_model(project_name)
    artifact_uri = latest_model_version.source
    artifact_base_dir = get_base_artifact_uri(artifact_uri)
    cfg = import_cfg_from_mlflow(artifact_base_dir,
                                 inference_cfg)

    # loaded_model = mlflow.pyfunc.load_model(artifact_uri)
    logger.info('Updating data directory () to the config'.format(data_dir))
    OmegaConf.update(cfg['hydra_cfg']['config'],
                     "DATA", {"DATA_SOURCE": {'FOLDER': {'DATA_DIR': data_dir}}})

    # Load the metamodel (i.e. Ensemble from Model Registry)
    download_dir = os.path.join(output_dir, 'mlflow_temp')
    os.makedirs(download_dir, exist_ok=True)
    model_uri, loaded_ensemble = download_registered_model(rmodel=rmodel,
                                                           latest_model_version=latest_model_version,
                                                           cfg=cfg,
                                                           artifact_base_dir=artifact_base_dir,
                                                           download_dir=download_dir,
                                                           model_download_method=model_download_method)

    mlflow_run = {
        'latest_model_version': latest_model_version,
        'artifact_base_dir': artifact_base_dir,
        'artifact_uri': artifact_uri,
        'model_uri': model_uri,
        'loaded_ensemble': loaded_ensemble
                  }

    cfg['mlflow_run'] = mlflow_run

    return cfg


def download_registered_model(rmodel: RegisteredModel,
                              latest_model_version: ModelVersion,
                              cfg: dict,
                              artifact_base_dir: str = None,
                              download_dir: str = None,
                              model_download_method: str = 'subdirs',
                              ensemble_name: str = 'dice-MINIVESS'):
    """
    https://mlflow.org/docs/latest/model-registry.html#fetching-an-mlflow-model-from-the-model-registry
    https://python.plainenglish.io/how-to-create-meta-model-using-mlflow-166aeb8666a8
    """

    if model_download_method == 'subdirs':
        loaded_ensemble, model_uri = (
            load_ensemble_from_mlflow_artifact_folder(artifact_base_dir=artifact_base_dir,
                                                      download_dir=download_dir,
                                                      cfg=cfg,
                                                      ensemble_name=ensemble_name))

    elif model_download_method == 'mlflow.pyfunc.load_model':
        raise NotImplementedError('This does not seem to work correctly, do some devel debugging')
        # https://stackoverflow.com/a/76347084/6412152
        # client = MlflowClient(mlflow.get_tracking_uri())
        # download_uri = client.get_model_version_download_uri(latest_model_version.name, latest_model_version.version)
        # model_uri_version = f"models:/{latest_model_version.name}/{latest_model_version.version}"
        # model_uri_stage = f"models:/{latest_model_version.name}/{latest_model_version.current_stage}"
        # model_uri = os.path.join(artifact_base_dir, 'MLmodel')
        # loaded_ensemble = create_model_ensemble_from_mlflow_models(model_uri=model_uri,
        #                                                            dst_path=download_dir)
    else:
        raise IOError('Unknown model_download_method = {}'.format(model_download_method))

    return model_uri, loaded_ensemble


def load_ensemble_from_mlflow_artifact_folder(artifact_base_dir: str,
                                              download_dir: str,
                                              cfg: dict,
                                              ensemble_name: str = 'dice-MINIVESS',
                                              device: str = 'cpu'):

    mlflow_uri = os.path.join(artifact_base_dir, define_ensemble_submodels_dir_name())
    local_path = mlflow.artifacts.download_artifacts(mlflow_uri, dst_path=download_dir)
    logger.info('Downloaded the ensemble submodels to local path = "{}"'.format(local_path))

    ensemble_subdir = os.path.join(local_path, ensemble_name)
    if not os.path.exists(ensemble_subdir):
        logger.error('Could not find the ensemble subdirectory = "{}"'.format(ensemble_subdir))
        raise IOError('Could not find the ensemble subdirectory = "{}"'.format(ensemble_subdir))

    model_paths = glob.glob(os.path.join(ensemble_subdir, '*'))
    models_of_ensemble = convert_model_path_list_to_dict(model_paths)
    logger.info('Found {} submodels in the ensemble subdirectory = "{}"'.format(len(model_paths), ensemble_subdir))
    if len(model_paths) == 0:
        logger.error('Could not find any submodels in the ensemble subdirectory = "{}"'.format(ensemble_subdir))
        raise IOError('Could not find any submodels in the ensemble subdirectory = "{}"'.format(ensemble_subdir))

    validation_params = cfg_key(cfg, 'hydra_cfg', 'config', 'VALIDATION', 'VALIDATION_PARAMS')
    ensemble_model = ModelEnsemble(models_of_ensemble=models_of_ensemble,
                                   validation_config=cfg_key(cfg, 'hydra_cfg', 'config', 'VALIDATION'),
                                   ensemble_params=cfg_key(cfg, 'hydra_cfg', 'config', 'ENSEMBLE', 'PARAMS'),
                                   validation_params=validation_params,
                                   device=device,
                                   eval_config=cfg_key(cfg, 'hydra_cfg', 'config', 'VALIDATION_BEST'),
                                   # TODO! this could be architecture-specific
                                   precision='AMP')

    return ensemble_model, mlflow_uri


def convert_model_path_list_to_dict(model_paths: list) -> dict:
    dict_out = {}
    for model_path in model_paths:
        model_name, _ = os.path.splitext(os.path.split(model_path)[1])
        dict_out[model_name] = model_path

    return dict_out


def get_base_artifact_uri(artifact_uri: str,
                          artifacts_dir_mlflow: str = 'artifacts'):
    # Examine later why is it like this as the project name is extra when
    # you are trying to access the actual artifact that you want
    base_dir, subdir = os.path.split(artifact_uri)
    if subdir == artifacts_dir_mlflow:
        return artifact_uri
    else:
        base_dir2, subdir2 = os.path.split(base_dir)
        if subdir2 == artifacts_dir_mlflow:
            return base_dir
        else:
            logger.error('More than one subdir in artifact_uri = {}'.format(artifact_uri))
            raise IOError('More than one subdir in artifact_uri = {}'.format(artifact_uri))


def import_cfg_from_mlflow(artifact_base_dir: str = None,
                           inference_cfg: str = None,
                           update_train_cfg_with_inference_cfg: bool = True):

    logger.info('Get training config ({}) from {}'.format(get_cfg_yaml_fname(), artifact_base_dir))
    try:
        local_path = mlflow.artifacts.download_artifacts(artifact_base_dir + f"/{get_cfg_yaml_fname()}")
    except Exception as e:
        logger.error('Problem downloading "{}" from MLflow Artifact Store! '
                     'e = {}'.format(get_cfg_yaml_fname(), e))
        raise IOError('Problem downloading "{}" from MLflow Artifact Store! '
                      'e = {}'.format(get_cfg_yaml_fname(), e))

    if update_train_cfg_with_inference_cfg:
        hydra_cfg = update_mlflow_cfg(local_path, inference_cfg)
    else:
        hydra_cfg = OmegaConf.load(local_path)

    logger.info('Get run parameters ({}) from {}'.format(get_run_params_yaml_fname(), artifact_base_dir))
    try:
        local_path = mlflow.artifacts.download_artifacts(artifact_base_dir + f"/{get_run_params_yaml_fname()}")
    except Exception as e:
        logger.error('Problem downloading "{}" from MLflow Artifact Store! '
                     'e = {}'.format(get_run_params_yaml_fname(), e))
        raise IOError('Problem downloading "{}" from MLflow Artifact Store! '
                      'e = {}'.format(get_cfg_yaml_fname(), e))
    run_params = OmegaConf.load(local_path)

    return {'hydra_cfg': hydra_cfg, 'run': run_params}


def update_mlflow_cfg(local_path: str, inference_cfg: str):

    config_dir = get_config_dir()

    # tempoprary copy of the MLflow config as the base (to match the training code)
    fname = os.path.split(local_path)[1]
    fname_wo_ext = os.path.splitext(fname)[0]

    try:
        shutil.copy(local_path, os.path.join(config_dir, fname))
    except Exception as e:
        logger.error('Failed to make the temp copy of the MLflow config '
                     '"{}" to "{}", e = {}'.format(fname, os.path.join(config_dir, fname), e))
        raise IOError('Failed to make the temp copy of the MLflow config '
                      '"{}" to "{}", e = {}'.format(fname, os.path.join(config_dir, fname), e))

    # combine the base from MLflow run with the inference config
    config = config_import_script(task_cfg_name=inference_cfg,
                                  parent_dir_string='..',
                                  parent_dir_string_defaults='..',
                                  job_name='inference_folder',
                                  base_cfg_name = fname_wo_ext)

    # remove the temporary copy of the MLflow config
    try:
        os.remove(os.path.join(config_dir, fname))
    except Exception as e:
        logger.warning('Failed to remove the temp file "{}", e = {}'.format(os.path.join(config_dir, fname), e))

    return config
