from loguru import logger

from src.log_ML.log_crossval import log_cv_results
from src.log_ML.log_epochs import log_epoch_for_tensorboard
from src.log_ML.results_utils import average_repeat_results, reorder_crossvalidation_results, compute_crossval_stats, \
    reorder_ensemble_crossvalidation_results, compute_crossval_ensemble_stats, get_cv_sample_stats_from_ensemble


def log_epoch_results(train_epoch_results, eval_epoch_results,
                      epoch, config, output_dir, output_artifacts):

    if 'epoch_level' not in list(output_artifacts.keys()):
        output_artifacts['epoch_level'] = {}

    output_artifacts = log_epoch_for_tensorboard(train_epoch_results, eval_epoch_results,
                                                 epoch, config, output_dir, output_artifacts)

    return output_artifacts

def log_n_epochs_results(train_results, eval_results, best_dict, output_artifacts, config):

    logger.debug('Placeholder for n epochs logging (i.e. submodel or single repeat training)')


def log_averaged_repeats(repeat_results: dict):

    # Average the repeat results (not ensembling per se yet, as we are averaging the metrics here, and not averaging
    # the predictions and then computing the metrics from the averaged predictions)
    averaged_results = average_repeat_results(repeat_results)

    # ADD the actual logging of the dict here, can be the same as for CV results
    logger.debug('Placeholder for averaged repeats')


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

    log_cv_results(cv_results=cv_results,
                   cv_ensemble_results=cv_ensemble_results,
                   config=config,
                   output_dir=output_dir)