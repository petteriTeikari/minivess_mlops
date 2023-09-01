import numpy as np

from src.inference.ensemble_utils import compute_stats_of_array_in_dict
from src.log_ML.log_utils import convert_value_to_numpy_array, compute_numpy_stats


def average_repeat_results(repeat_results):

    # exploit the same function written for cv results, and a dummy fold to the repeats
    dummy_fold_results = {'dummy_fold': repeat_results}
    reordered_dummy_fold_results = reorder_crossvalidation_results(dummy_fold_results)
    averaged_results = compute_crossval_stats(reordered_dummy_fold_results)

    return averaged_results


def reorder_crossvalidation_results(fold_results: dict):

    res_out = {}

    no_of_folds = len(fold_results)
    for f, fold_key in enumerate(fold_results):
        fold_result = fold_results[fold_key]
        no_of_repeats = len(fold_result)  # same as number of submodels in an inference

        for r, repeat_key in enumerate(fold_result):
            repeat_results = fold_result[repeat_key]
            repeat_best = repeat_results['best_dict']

            for d, ds_name in enumerate(repeat_best):
                if ds_name not in res_out.keys():
                    res_out[ds_name] = {}

                for m, metric in enumerate(repeat_best[ds_name]):
                    if metric not in res_out[ds_name].keys():
                        res_out[ds_name][metric] = {}
                    repeat_best_eval = repeat_best[ds_name][metric]['eval_epoch_results']

                    for s, split in enumerate(repeat_best_eval):
                        if split not in res_out[ds_name][metric].keys():
                            res_out[ds_name][metric][split] = {}

                        for d_e, ds_name_eval in enumerate(repeat_best_eval[split]):
                            if ds_name_eval not in res_out[ds_name][metric][split].keys():
                                res_out[ds_name][metric][split][ds_name_eval] = {}
                            eval_metrics = repeat_best_eval[split][ds_name_eval]

                            for t, var_type in enumerate(eval_metrics):
                                if var_type not in res_out[ds_name][metric][split][ds_name_eval].keys():
                                    res_out[ds_name][metric][split][ds_name_eval][var_type] = {}

                                for v, var_name in enumerate(eval_metrics[var_type]):
                                    value_in = convert_value_to_numpy_array(eval_metrics[var_type][var_name])
                                    # value_in will have a shape of (1,) for scalars and will be aggregating them
                                    # so that you will have (no_folds, no_repeats) np.arrays in the rearranged dict
                                    if var_name not in res_out[ds_name][metric][split][ds_name_eval][
                                        var_type].keys():
                                        value_array = np.zeros((no_of_folds, no_of_repeats))
                                        res_out[ds_name][metric][split][ds_name_eval][var_type][
                                            var_name] = value_array

                                    # res_out_tmp = res_out[ds_name][metric][split][ds_name_eval][var_type][var_name]
                                    res_out[ds_name][metric][split][ds_name_eval][var_type][var_name][
                                        f, r] = value_in

    return res_out


def compute_crossval_stats(fold_results_reordered: dict):

    res_out = {}

    for d, ds_name in enumerate(fold_results_reordered):
        if ds_name not in res_out.keys():
            res_out[ds_name] = {}

        for m, metric in enumerate(fold_results_reordered[ds_name]):
            if metric not in res_out[ds_name].keys():
                res_out[ds_name][metric] = {}
            best_metric = fold_results_reordered[ds_name][metric]

            for s, split in enumerate(best_metric):
                if split not in res_out[ds_name][metric].keys():
                    res_out[ds_name][metric][split] = {}

                for d_e, ds_name_eval in enumerate(best_metric[split]):
                    if ds_name_eval not in res_out[ds_name][metric][split].keys():
                        res_out[ds_name][metric][split][ds_name_eval] = {}
                    eval_metrics = best_metric[split][ds_name_eval]

                    for t, var_type in enumerate(eval_metrics):
                        if var_type not in res_out[ds_name][metric][split][ds_name_eval].keys():
                            res_out[ds_name][metric][split][ds_name_eval][var_type] = {}

                        for v, var_name in enumerate(eval_metrics[var_type]):
                            value_array_in = eval_metrics[var_type][var_name]
                            res_out[ds_name][metric][split][ds_name_eval][var_type][var_name] = (
                                compute_numpy_stats(value_array_in))

    return res_out


def compute_crossval_ensemble_stats(ensembled_results_reordered: dict):

    stats_out = {}
    for s, split in enumerate(ensembled_results_reordered):
        stats_out[split] = {}
        for d, dataset in enumerate(ensembled_results_reordered[split]):
            stats_out[split][dataset] = {}
            for m, tracked_metric in enumerate(ensembled_results_reordered[split][dataset]):
                stats_out[split][dataset][tracked_metric] = {}
                for m_e, metric in enumerate(ensembled_results_reordered[split][dataset][tracked_metric]):
                    stats_out[split][dataset][tracked_metric][metric] = {}
                    for s, stat in enumerate(ensembled_results_reordered[split][dataset][tracked_metric][metric]):
                        # the results might be confusing at first, as easiest just to loop with the same operations
                        # you get mean of n for example that might be quite useless. And mean of mean with dice just
                        # means that "1st mean" came from all the samples in the dataloader, and the "2nd mean" is
                        # the mean of all the folds
                        values = ensembled_results_reordered[split][dataset][tracked_metric][metric][stat]
                        stats_dict = compute_stats_of_array_in_dict(values)
                        stats_out[split][dataset][tracked_metric][metric][stat] = stats_dict

    return stats_out


def reorder_ensemble_crossvalidation_results(ensembled_results,
                                             ensemble_stats_key: str = 'stats'):

    # ensembled_results['fold1'] = ensembled_results['fold0']  # debug extra fold
    no_of_folds = len(ensembled_results)
    stats_out = {}

    for f, fold_key in enumerate(ensembled_results):
        for s, split_name in enumerate(ensembled_results[fold_key]):
            if split_name not in stats_out:
                stats_out[split_name] = {}
            stats_per_fold = ensembled_results[fold_key][split_name][ensemble_stats_key]

            for d, ds_name in enumerate(stats_per_fold):
                if ds_name not in stats_out[split_name]:
                    stats_out[split_name][ds_name] = {}
                for m, tracked_metric in enumerate(stats_per_fold[ds_name]):
                    if tracked_metric not in stats_out[split_name][ds_name]:
                        stats_out[split_name][ds_name][tracked_metric] = {}
                    metrics_per_tracked_metric = stats_per_fold[ds_name][tracked_metric]['metrics']

                    for m_e, metric in enumerate(metrics_per_tracked_metric):
                        if metric not in stats_out[split_name][ds_name][tracked_metric]:
                            stats_out[split_name][ds_name][tracked_metric][metric] = {}
                        for s, stat_key in enumerate(metrics_per_tracked_metric[metric]):
                            stat_value = metrics_per_tracked_metric[metric][stat_key]
                            stat_array = np.expand_dims(np.array(stat_value), axis=0)
                            if stat_key not in stats_out[split_name][ds_name][tracked_metric][metric]:
                                stats_out[split_name][ds_name][tracked_metric][metric][stat_key] = stat_array
                            else:
                                stats_out[split_name][ds_name][tracked_metric][metric][stat_key] = (
                                    np.concatenate((stats_out[split_name][ds_name][tracked_metric][metric][stat_key], 
                                                   stat_array), axis=0))

    return stats_out


def get_cv_sample_stats_from_ensemble(ensembled_results,
                                      sample_key: str = 'samples',
                                      split_key: str = 'TEST'):
    """
    We have here sample-level performance per fold.
    If you want to see how specific sample in "TEST" split performs across the folds (you could later do for VAL/TRAIN,
    but these would require more code as you don't have the same data then on these splits across the folds)

    :param ensembled_results:
    :param sample_key:
    :param split_key:
    :return:
    """

    no_of_folds = len(ensembled_results)
    stats_out = {}

    for f, fold_key in enumerate(ensembled_results):
        if split_key in ensembled_results[fold_key]:
            if split_key not in stats_out:
                stats_out[split_key] = {}
            sample_stats_per_fold = ensembled_results[fold_key][split_key][sample_key]
            # TO BE CONTINUED FROM HERE

    stats_out = 'not_implemented_yet'
    return stats_out