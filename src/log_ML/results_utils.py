import numpy as np

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