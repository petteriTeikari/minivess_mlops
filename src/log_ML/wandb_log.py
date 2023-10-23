import os
import glob

import mlflow
import omegaconf
from loguru import logger
import wandb

from src.log_ML.log_config import log_config_artifacts
from src.log_ML.log_utils import get_number_of_steps_from_repeat_results
from src.log_ML.mlflow_log import mlflow_cv_artifacts


def log_wandb_repeat_results(fold_results: dict,
                             output_dir: str,
                             config: dict,
                             log_models_with_repeat: bool = False):

    logger.info('Logging repeat-wise results to WANDB (local_dir = {})'.format(output_dir))
    for f, fold_name in enumerate(fold_results):
        for a, archi_name in enumerate(fold_results[fold_name]):
            for r, repeat_name in enumerate(fold_results[fold_name][archi_name]):

                # In theory, you could want to separate the fold and repeat, but there is
                # no additional field in WANDB for this type of hierarchy
                # use ensemble_name function?
                log_name = fold_name + '_' + archi_name + '_' + repeat_name
                try:
                    wandb_run = wandb_init_wrapper(project=config['ARGS']['project_name'],
                                                   name=log_name,
                                                   group=config['run']['hyperparam_name'],
                                                   job_type='repeat',
                                                   param_conf=config['config'],
                                                   dir_out=output_dir,
                                                   tags=['tag1', 'tag2'])
                except Exception as e:
                    raise Exception('Problem with initializing Weights and Biases, e = {}'.format(e))

                wandb_log_repeat(wandb_run, fold_name, repeat_name, log_name,
                                 results=fold_results[fold_name][archi_name][repeat_name], config=config,
                                 log_models_with_repeat=log_models_with_repeat)
                wandb.finish()


def wandb_log_repeat(wandb_run, fold_name: str, repeat_name: str, log_name: str,
                     results: dict, config: dict, log_models_with_repeat: bool = False):

    # Log the training curves from the repeat
    wandb_log_metrics_of_repeat(wandb_run, results)

    # Log models to Artifact Store (and Model Registry if you want/need to use)
    if log_models_with_repeat:
        artifact_type = 'model'  # This is "reserved name", and if you want to use Model Registry, keep this naming
        artifact = wandb.Artifact(name=log_name, type=artifact_type)
        wandb_log_models_from_model_dir(wandb_run, artifact, model_path=results['output_artifacts']['model_dir'])
        wandb_run.log_artifact(artifact)
    else:
        # Saves some disk space and is conceptually simpler if you only log the models
        # once and at the CV ensembling part you will have all the models at hand (all folds, all repeats)
        logger.info('Skipping model artifact logging at repeat-level logging! Logging later with CV ensemble')

    # If you want to add some artifacts dir per repeat (e.g. custom plots), here is your chance
    # artifact.add_dir()


def wandb_init_wrapper(project: str = None,
                       name: str = None,
                       group: str = None,
                       dir_out: str = None,
                       job_type: str = 'repeat',
                       param_conf: dict = None,
                       notes: str = None,
                       tags=None) -> wandb.sdk.wandb_run.Run:

    os.makedirs(dir, exist_ok=True)

    # Remember that the config dictionary might be actually an OmegaConf object, and we need to
    # convert it back to ordinary dictionary if you want to dump it as a parameter dictionary to WANDB
    if isinstance(param_conf, omegaconf.dictconfig.DictConfig):
        logger.info('Convert parameter dictionary from OmegaConf to Dict')
        param_conf = dict(param_conf)

    # https://docs.wandb.ai/ref/python/init
    # logger.info('Initialize Weights and Biases logging, \n'
    #             'project = "{}"'.format(project))
    try:
        wandb_run = wandb.init(project=project,
                               job_type=job_type,
                               group=group,
                               name=name,
                               dir=dir,
                               notes=notes,
                               tags=tags,
                               config=param_conf,
                               save_code=True)
    except Exception as e:
        raise IOError('Problem initializing WANDB, e = {}'.format(e))

    return wandb_run


def log_wandb_ensemble_results(ensembled_results: dict,
                               output_dir: str,
                               config: dict):

    logger.info('Logging ensemble-wise results to WANDB')
    for f, fold_name in enumerate(ensembled_results):

        log_name = fold_name
        try:
            wandb_run = wandb_init_wrapper(project=config['ARGS']['project_name'],
                                           name=log_name,
                                           group=config['run']['hyperparam_name'],
                                           job_type='ensemble',
                                           param_conf=config['config'],
                                           dir_out=output_dir,
                                           tags=['tag1', 'tag2'])
        except Exception as e:
            raise Exception('Problem with initializing Weights and Biases, e = {}'.format(e))

        # Log the metrics
        # TO-OPTIMIZE, you could think of using the same function really for both WANDB and MLflow
        wandb_log_ensemble_per_fold(wandb_run, fold_name, log_name,
                                    results=ensembled_results[fold_name], config=config)

        # Log the artifacts, as in all the custom plots, .csv., .json you might have
        # created during the training process
        dir_out = config['run']['ensemble_artifacts'][fold_name]
        logger.info('Logging the ensemble output directory to WANDB: {}'.format(dir_out))
        artifact = wandb.Artifact(name=log_name, type='artifacts')
        artifact.add_dir(local_path=dir_out, name='ensemble_artifacts')
        wandb_run.log_artifact(artifact)

        wandb.finish()


def log_crossval_res(cv_results: dict,
                     cv_ensemble_results: dict,
                     ensembled_results: dict,
                     fold_results: dict,
                     experim_dataloaders: dict,
                     cv_averaged_output_dir: str,
                     cv_ensembled_output_dir: str,
                     output_dir: str,
                     logging_services: list,
                     config: dict):

    if len(logging_services) == 0:
        logger.warning('No logging (Experiment tracking such as MLflow or WANDB) '
                       'services defined, skipping the logging!')
        return None
    else:
        logger.info('Logging AVERAGED Cross-Validation results')
        log_cv_results(cv_results=cv_results,
                       cv_dir_out=cv_averaged_output_dir,
                       config=config,
                       logging_services=logging_services,
                       output_dir=output_dir)

        logger.info('Logging ENSEMBLED Cross-Validation results to WANDB')
        model_paths = log_cv_ensemble_results(cv_ensemble_results=cv_ensemble_results,
                                              ensembled_results=ensembled_results,
                                              cv_dir_out=cv_ensembled_output_dir,
                                              fold_results=fold_results,
                                              experim_dataloaders=experim_dataloaders,
                                              config=config,
                                              logging_services=logging_services,
                                              output_dir=output_dir)

    return model_paths


def log_cv_results(cv_results: dict,
                   cv_dir_out: str,
                   config: dict,
                   logging_services: list,
                   output_dir: str,
                   stat_keys_to_reject: tuple = ('n',),
                   metrics_to_metadata: tuple = ('time_',),  # quick n dirty method to keep times from metrics
                   var_types: tuple = ('scalars', 'metadata_scalars')):

    log_name = 'CV_averaged'
    if 'WANDB' in logging_services:
        try:
            wandb_run = wandb_init_wrapper(project=config['ARGS']['project_name'],
                                           name=log_name,
                                           group=config['run']['hyperparam_name'],
                                           job_type='CV',
                                           param_conf=config['config'],
                                           dir_out=output_dir,
                                           tags=['tag1', 'tag2'])
        except Exception as e:
            raise Exception('Problem with initializing Weights and Biases, e = {}'.format(e))

    # Log the scalar metrics
    for dataset in cv_results:
        for tracked_metric in cv_results[dataset]:
            for split in cv_results[dataset][tracked_metric]:
                # Double-check if the double dataset definition is needed
                for ds_eval in cv_results[dataset][tracked_metric][split]:
                    for var_type in cv_results[dataset][tracked_metric][split][ds_eval]:
                        if var_type in var_types:
                            for var in cv_results[dataset][tracked_metric][split][ds_eval][var_type]:
                                for stat_key in cv_results[dataset][tracked_metric][split][ds_eval][var_type][var]:
                                    if stat_key not in stat_keys_to_reject:
                                        v = cv_results[dataset][tracked_metric][split][ds_eval][var_type][var][stat_key]
                                        metric_name = 'CV_{}/{}/{}_{}'.format(split, dataset, var, stat_key)

                                        log_cv_metric(logging_services=logging_services,
                                                      metric_name=metric_name,
                                                      value=v,
                                                      metrics_to_metadata=metrics_to_metadata)

                                        # TOADD! Add the main metric definition here as well
                                        # NOTE!  ['MLflow', 'WANDB'] | "CV_VAL/MINIVESS/time_e , reject the timing?
                                        #        or group at least to some metadata window?

    # Log the artifact dir
    if 'WANDB' in logging_services:
        logger.info('WANDB | Logging the directory {} as an artifact'.format(cv_dir_out))
        artifact = wandb.Artifact(name=log_name, type='artifacts')
        artifact.add_dir(local_path=cv_dir_out, name='CV_artifacts')
        wandb_run.log_artifact(artifact)
        wandb.finish()

    if 'MLflow' in logging_services:
        mlflow_cv_artifacts(log_name=log_name, local_artifacts_dir=cv_dir_out)


def log_cv_metric(logging_services: list, metric_name: str, value,
                  metrics_to_metadata: tuple = None,
                  log_this_metric: bool = True,
                  reject_metadata: bool = True):

    if metrics_to_metadata is not None:
        for wildcard in metrics_to_metadata:
            if wildcard in metric_name:
                if reject_metadata:
                    log_this_metric = False
                else:
                    metric_name = 'metadata_' + metric_name

    if log_this_metric:
        logger.info('{} | "{}": {:.3f}'.format(logging_services, metric_name, value))
        if 'WANDB' in logging_services:
            wandb.log({metric_name: value}, step=0)

        if 'MLflow' in logging_services:
            mlflow.log_metric(metric_name, value)
    else:
        logger.debug('SKIP THIS: {} | "{}": {:.3f}'.format(logging_services, metric_name, value))


def log_cv_ensemble_results(cv_ensemble_results: dict,
                            ensembled_results: dict,
                            cv_dir_out: str,
                            fold_results: dict,
                            experim_dataloaders: dict,
                            config: dict,
                            logging_services: list,
                            output_dir: str,
                            stat_keys_to_reject: tuple = ('n', ),
                            stat_key2: str = 'mean'):

    log_name = 'CV_ensembled'
    if 'WANDB' in logging_services:
        try:
            wandb_run = wandb_init_wrapper(project=config['ARGS']['project_name'],
                                           name=log_name,
                                           group=config['run']['hyperparam_name'],
                                           job_type='CV_ENSEMBLE',
                                           param_conf=config['config'],
                                           dir_out=output_dir,
                                           tags=['tag1', 'tag2'])
        except Exception as e:
            raise Exception('Problem with initializing Weights and Biases, e = {}'.format(e))
    else:
        wandb_run = None

    logger.info('ENSEMBLED Cross-Validation results | Metrics')
    for split in cv_ensemble_results:
        for ensemble_name in cv_ensemble_results[split]:
            for metric in cv_ensemble_results[split][ensemble_name]:
                for stat_key in cv_ensemble_results[split][ensemble_name][metric]:
                    if stat_key not in stat_keys_to_reject:

                        # CHECK THIS, IF THIS GOES CORRECTLY?
                        stat_dict = cv_ensemble_results[split][ensemble_name][metric][stat_key]
                        value = stat_dict[stat_key2]
                        metric_name = 'CV-ENSEMBLE_{}/{}/{}_{}'.format(split, ensemble_name, metric, stat_key)

                        log_cv_ensemble_metric(logging_services=logging_services,
                                               metric_name=metric_name,
                                               value=value)

                        # TOADD! Add the main metric definition here as well
                        # NOTE! If only one fold, no need to log the stdev of 0

    # Log the artifacts, config, log and the model(s) to Model Registry
    model_paths = log_config_artifacts(log_name=log_name,
                                       cv_dir_out=cv_dir_out,
                                       output_dir=output_dir,
                                       config=config,
                                       fold_results=fold_results,
                                       ensembled_results=ensembled_results,
                                       cv_ensemble_results=cv_ensemble_results,
                                       experim_dataloaders=experim_dataloaders,
                                       logging_services=logging_services,
                                       wandb_run=wandb_run)

    if 'WANDB' in logging_services:
        wandb.finish()  # end of WANDB logging here
        logger.info('Done with the WANDB Logging!')

    return model_paths


def log_cv_ensemble_metric(logging_services: list, metric_name: str, value):

    logger.info('{} | "{}": {:.3f}'.format(logging_services, metric_name, value))
    if 'WANDB' in logging_services:
        wandb.log({metric_name: value}, step=0)

    if 'MLflow' in logging_services:
        mlflow.log_metric(metric_name, value)


def wandb_log_metrics_of_repeat(wandb_run, results):

    no_train_batches = get_number_of_steps_from_repeat_results(results, result_type='train_results')
    no_train_epochs = get_number_of_steps_from_repeat_results(results, result_type='eval_results')
    wandb_log_array(res=results['eval_results'], result_type='eval_results', no_steps=no_train_epochs)


def wandb_log_models_from_model_dir(wandb_run, artifact, model_path: str, model_ext: str = '*.pth'):

    model_files = glob.glob(os.path.join(model_path, model_ext))
    for model_file in model_files:
        artifact.add_file(model_file)


def wandb_log_array(res: dict, result_type: str = 'eval_results', no_steps: int = None,
                    var_types: tuple = ('scalars', 'metadata_scalars')):

    # Note! Wandb require that you log each metric per step, in contrast to maybe more
    # intuitive way of looping through the steps of each metric
    for step in range(no_steps):
        for split in res:
            for dataset in res[split]:
                for var_type in res[split][dataset]:
                    if var_type in var_types:
                        if len(res[split][dataset][var_type]) > 0:
                            for var in res[split][dataset][var_type]:
                                if var_type == 'metadata_scalars':
                                    metric_name = 'METADATA/{}/{}/{}'.format(split, dataset, var)
                                else:
                                    metric_name = '{}/{}/{}'.format(split, dataset, var)
                                value_in = res[split][dataset][var_type][var][step]
                                # print(step, split, dataset, var_type, var, metric_name, value_in)
                                wandb.log({metric_name: value_in}, step=step)


def wandb_log_ensemble_per_fold(wandb_run, fold_name: str, log_name: str,
                                results: dict, config: dict,
                                stats_key: str = 'stats',
                                metrics_key: str = 'metrics',
                                var_keys: tuple = ('mean', 'var')):

    for split in results:
        for ensemble_name in results[split]:
            ensemble_stats = results[split][ensemble_name][stats_key]
            ensemble_metrics = results[split][ensemble_name][stats_key][metrics_key]
            for metric in ensemble_metrics:
                value_dict = ensemble_metrics[metric]
                for var_key in value_dict:
                    if var_key in var_keys:
                        value = value_dict[var_key]
                        metric_name = 'ENSEMBLE_{}/{}/{}_{}'.format(split, ensemble_name, metric, var_key)
                        logger.info('WANDB | "{}": {:.3f}'.format(metric_name, value))
                        wandb.log({metric_name: value}, step=0)
