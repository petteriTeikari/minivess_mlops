import mlflow
from loguru import logger

from src.inference.ensemble_main import reinference_dataloaders
from src.log_ML.log_crossval import log_cv_results
from src.log_ML.log_epochs import log_epoch_for_tensorboard
from src.log_ML.results_utils import average_repeat_results, reorder_crossvalidation_results, compute_crossval_stats, \
    reorder_ensemble_crossvalidation_results, compute_crossval_ensemble_stats, get_cv_sample_stats_from_ensemble, \
    get_best_repeat_result
from src.log_ML.wandb_log import log_wandb_repeat_results, log_ensemble_results


def log_epoch_results(train_epoch_results, eval_epoch_results,
                      epoch, config, output_dir, output_artifacts):

    if 'epoch_level' not in list(output_artifacts.keys()):
        output_artifacts['epoch_level'] = {}

    output_artifacts = log_epoch_for_tensorboard(train_epoch_results, eval_epoch_results,
                                                 epoch, config, output_dir, output_artifacts)

    return output_artifacts

def log_n_epochs_results(train_results, eval_results, best_dict, output_artifacts, config,
                         repeat_idx: int, fold_name: str, repeat_name: str):

    logger.debug('Placeholder for n epochs logging (i.e. submodel or single repeat training)')


def log_averaged_and_best_repeats(repeat_results: dict,
                                  fold_name: str,
                                  config: dict,
                                  dataloaders: dict,
                                  device):

    # Average the repeat results (not ensembling per se yet, as we are averaging the metrics here, and not averaging
    # the predictions and then computing the metrics from the averaged predictions)
    averaged_results = average_repeat_results(repeat_results)

    # You might want to compare single model performance to ensemble and quantify the improvement from
    # added computational cost from ensembling over single repeat (submodel)
    best_repeat_dicts = get_best_repeat_result(repeat_results)

    # Log the metric results of the best repeat out of n repeats
    log_best_repeats(best_repeat_dicts=best_repeat_dicts,
                     config=config,
                     service='MLflow')

    # Re-inference the dataloader with the best model(s) of the best repeat
    # e.g. you want to have Hausdorff distance here that you thought to be too heavy to compute while training
    best_repeat_metrics = reinference_dataloaders(input_dict=best_repeat_dicts,
                                                  config=config,
                                                  artifacts_output_dir=config['run']['output_experiment_dir'],
                                                  dataloaders=dataloaders,
                                                  device=device,
                                                  model_scheme='best_repeats')

    # log best repeat metrics here to MLflow/WANDB
    log_best_reinference_metrics(best_repeat_metrics=best_repeat_metrics,
                                 config=config)


def log_best_repeats(best_repeat_dicts: dict, config: dict,
                     splits: tuple = ('VAL', 'TEST'),
                     service: str = 'MLflow'):

    logger.info('Logging (MLflow) the metrics obtained from best repeat')
    for dataset_train in best_repeat_dicts:
        for tracked_metric in best_repeat_dicts[dataset_train]:
            for split in best_repeat_dicts[dataset_train][tracked_metric]:
                if split in splits:
                    for dataset_eval in best_repeat_dicts[dataset_train][tracked_metric][split]:
                        for metric in best_repeat_dicts[dataset_train][tracked_metric][split][dataset_eval]:
                            best_repeat = best_repeat_dicts[dataset_train][tracked_metric][split][dataset_eval][metric]
                            metric_name = 'bestRepeat_{}_{}/{}/{}/{}'.format(metric, split, dataset_train,
                                                                             tracked_metric, dataset_eval)
                            metric_value = best_repeat['best_value']
                            logger.info('"{}": {:.3f}'.format(metric_name, metric_value))

                            if service == 'MLflow':
                                mlflow.log_metric(metric_name, metric_value)
                            else:
                                raise NotImplementedError('Unknown Experiment Tracking service = "{}"'.format(service))


def log_crossvalidation_results(fold_results: dict,
                                ensembled_results: dict,
                                config: dict,
                                output_dir: str):

    # Reorder CV results and compute the stats, i.e. mean of 5 folds and 5 repeats per fold (n = 25)
    # by default, computes for all the splits, but you most likely are the most interested in the TEST
    fold_results_reordered = reorder_crossvalidation_results(fold_results)
    cv_results = compute_crossval_stats(fold_results_reordered)

    ensembled_results_reordered = reorder_ensemble_crossvalidation_results(ensembled_results)
    cv_ensemble_results = compute_crossval_ensemble_stats(ensembled_results_reordered)
    sample_cv_results = get_cv_sample_stats_from_ensemble(ensembled_results)

    log_wandb_repeat_results(fold_results=fold_results,
                             output_dir=config['run']['output_base_dir'],
                             config=config)

    log_ensemble_results(ensembled_results=ensembled_results,
                         output_dir=config['run']['output_base_dir'],
                         config=config)

    logged_model_paths = log_cv_results(cv_results=cv_results,
                                        cv_ensemble_results=cv_ensemble_results,
                                        fold_results=fold_results,
                                        config=config,
                                        output_dir=config['run']['output_base_dir'],
                                        cv_averaged_output_dir=config['run']['cross_validation_averaged'],
                                        cv_ensembled_output_dir=config['run']['cross_validation_ensembled'])

    logger.info('Done with the WANDB Logging!')
    mlflow.end_run()
    logger.info('Done with the MLflow Logging!')

    return logged_model_paths


def log_best_reinference_metrics(best_repeat_metrics: dict,
                                 config: dict):

    a = 'placeholder'
