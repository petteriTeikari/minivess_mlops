import os
from datetime import datetime
import glob

import omegaconf
from loguru import logger
import wandb

from src.log_ML.log_utils import get_number_of_steps_from_repeat_results, write_config_as_yaml


def log_wandb_repeat_results(fold_results: dict,
                             output_dir: str,
                             config: dict,
                             log_models_with_repeat: bool = False):

    logger.info('Logging repeat-wise results to WANDB (local_dir = {})'.format(output_dir))
    for f, fold_name in enumerate(fold_results):
        for r, repeat_name in enumerate(fold_results[fold_name]):

            # In theory, you could want to separate the fold and repeat, but there is
            # no additional field in WANDB for this type of hierarchy
            log_name = fold_name + '_' + repeat_name
            try:
                wandb_run = wandb_init_wrapper(project=config['ARGS']['project_name'],
                                               name=log_name,
                                               group=config['run']['hyperparam_name'],
                                               job_type='repeat',
                                               param_conf=config['config'],
                                               dir=output_dir,
                                               tags=['tag1', 'tag2'])
            except Exception as e:
                raise Exception('Problem with initializing Weights and Biases, e = {}'.format(e))

            wandb_log_repeat(wandb_run, fold_name, repeat_name, log_name,
                             results=fold_results[fold_name][repeat_name], config=config,
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
        wandb_log_models_from_model_dir(wandb_run, artifact, model_path = results['output_artifacts']['model_dir'])
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
                       dir: str = None,
                       job_type: str = 'repeat',
                       param_conf: dict = None,
                       notes: str = None,
                       tags = None) -> wandb.sdk.wandb_run.Run:

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
        wandb_run = wandb.init(project = project,
                               job_type = job_type,
                               group = group,
                               name = name,
                               dir = dir,
                               notes = notes,
                               tags = tags,
                               config = param_conf,
                               save_code=True)
    except Exception as e:
        raise IOError('Problem initializing WANDB, e = {}'.format(e))

    return wandb_run


def log_ensemble_results(ensembled_results: dict,
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
                                           dir=output_dir,
                                           tags=['tag1', 'tag2'])
        except Exception as e:
            raise Exception('Problem with initializing Weights and Biases, e = {}'.format(e))

        # Log the metrics
        wandb_log_ensemble_per_fold(wandb_run, fold_name, log_name,
                                    results=ensembled_results[fold_name], config=config)

        # Log the artifacts
        dir_out = config['run']['ensemble_artifacts'][fold_name]
        artifact = wandb.Artifact(name=log_name, type='artifacts')
        artifact.add_dir(local_path=dir_out, name='ensemble_artifacts')
        wandb_run.log_artifact(artifact)

        wandb.finish()


def wandb_log_crossval(cv_results: dict,
                       cv_ensemble_results: dict,
                       fold_results: dict,
                       cv_averaged_output_dir: str,
                       cv_ensembled_output_dir: str,
                       output_dir: str,
                       config: dict):

    logger.info('Logging AVERAGED Cross-Validation results to WANDB')
    wandb_log_cv_results(cv_results=cv_results,
                         cv_dir_out=cv_averaged_output_dir,
                         config=config,
                         output_dir=output_dir)

    logger.info('Logging ENSEMBLED Cross-Validation results to WANDB')
    model_paths = wandb_log_cv_ensemble_results(cv_ensemble_results=cv_ensemble_results,
                                                cv_dir_out=cv_ensembled_output_dir,
                                                fold_results=fold_results,
                                                config=config,
                                                output_dir=output_dir)

    return model_paths


def wandb_log_cv_results(cv_results: dict, cv_dir_out: str, config: dict, output_dir: str,
                         stat_keys_to_reject: tuple = ('n'),
                         var_types: tuple = ('scalars', 'metadata_scalars')):

    log_name = 'CV_averaged'
    try:
        wandb_run = wandb_init_wrapper(project=config['ARGS']['project_name'],
                                       name=log_name,
                                       group=config['run']['hyperparam_name'],
                                       job_type='CV',
                                       param_conf=config['config'],
                                       dir=output_dir,
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
                                        wandb.log({metric_name: v}, step=0)

    # Log the artifact dir
    artifact = wandb.Artifact(name=log_name, type='artifacts')
    artifact.add_dir(local_path=cv_dir_out, name='CV_artifacts')
    wandb_run.log_artifact(artifact)

    wandb.finish()


def wandb_log_cv_ensemble_results(cv_ensemble_results: dict,
                                  cv_dir_out: str,
                                  fold_results: dict,
                                  config: dict,
                                  output_dir: str,
                                  stat_keys_to_reject: tuple = ('n')):

    log_name = 'CV_ensembled'
    try:
        wandb_run = wandb_init_wrapper(project=config['ARGS']['project_name'],
                                       name=log_name,
                                       group=config['run']['hyperparam_name'],
                                       job_type='CV_ENSEMBLE',
                                       param_conf=config['config'],
                                       dir=output_dir,
                                       tags=['tag1', 'tag2'])
    except Exception as e:
        raise Exception('Problem with initializing Weights and Biases, e = {}'.format(e))

    logger.info('ENSEMBLED Cross-Validation results | Metrics')
    for split in cv_ensemble_results:
        for dataset in cv_ensemble_results[split]:
            for tracked_metric in cv_ensemble_results[split][dataset]:
                for metric in cv_ensemble_results[split][dataset][tracked_metric]:
                    for stat_key in cv_ensemble_results[split][dataset][tracked_metric][metric]:
                        if stat_key not in stat_keys_to_reject:
                            value = cv_ensemble_results[split][dataset][tracked_metric][metric][stat_key]
                            metric_name = 'CV-ENSEMBLE_{}/{}/{}_{}'.format(split, dataset, metric, stat_key)
                            wandb.log({metric_name: value}, step=0)

    # Log the artifact dir
    logger.info('ENSEMBLED Cross-Validation results | Artifacts directory')
    artifact_dir = wandb.Artifact(name=log_name, type='artifacts')
    artifact_dir.add_dir(local_path=cv_dir_out, name='CV-ENSEMBLE_artifacts')
    wandb_run.log_artifact(artifact_dir)

    # Log all the models from all the folds and all the repeats to the Artifact Store
    # and these are accessible to Model Registry as well
    logger.info('ENSEMBLED Cross-Validation results | Models to Model Registry')
    model_paths = wandb_log_models_to_artifact_store_from_fold_results(fold_results=fold_results,
                                                                       log_name=log_name,
                                                                       wandb_run=wandb_run)

    # HERE, log the config as .yaml file back to disk
    logger.info('ENSEMBLED Cross-Validation results | Config as YAML')
    path_out = write_config_as_yaml(config=config, dir_out=output_dir)
    artifact_cfg = wandb.Artifact(name='config', type='config')
    artifact_cfg.add_file(path_out)
    wandb_run.log_artifact(artifact_cfg)

    #
    logger.info('TODO! ENSEMBLED Cross-Validation results | Loguru log to WANDB')

    wandb.finish()

    return model_paths


def wandb_log_models_to_artifact_store_from_fold_results(fold_results: dict,
                                                         log_name: str,
                                                         wandb_run):
    model_paths = {}
    n_models = 0

    artifact_type = 'model'  # This is "reserved name", and if you want to use Model Registry, keep this naming

    for fold_name in fold_results:
        model_paths[fold_name] = {}
        for repeat_name in fold_results[fold_name]:
            model_paths[fold_name][repeat_name] = {}
            best_dict = fold_results[fold_name][repeat_name]['best_dict']
            for ds in best_dict:
                model_paths[fold_name][repeat_name][ds] = {}
                for tracked_metric in best_dict[ds]:
                    model_path = best_dict[ds][tracked_metric]['model']['model_path']
                    model_paths[fold_name][repeat_name][ds][tracked_metric] = model_path
                    n_models += 1

                    artifact_name = '{}_{}'.format(fold_name, repeat_name)
                    artifact_model = wandb.Artifact(name=artifact_name, type=artifact_type)
                    artifact_model.add_file(model_path)
                    wandb_run.log_artifact(artifact_model)


    return model_paths


def wandb_log_metrics_of_repeat(wandb_run, results):

    no_train_batches = get_number_of_steps_from_repeat_results(results, result_type = 'train_results')
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
        for dataset in results[split][stats_key]:
            for tracked_metric in results[split][stats_key][dataset]:
                for metric in results[split][stats_key][dataset][tracked_metric][metrics_key]:
                    value_dict = results[split][stats_key][dataset][tracked_metric][metrics_key][metric]
                    for var_key in value_dict:
                        if var_key in var_keys:
                            value = value_dict[var_key]
                            metric_name = 'ENSEMBLE_{}/{}/{}_{}'.format(split, dataset, metric, var_key)
                            # print(split, dataset, tracked_metric, metric, var_key, value, metric_name)
                            wandb.log({metric_name: value}, step=0)

