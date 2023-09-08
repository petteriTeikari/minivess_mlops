import os
from loguru import logger

import numpy as np


def add_sample_results_to_ensemble_results(inf_res: dict,
                                           sample_res: dict):

    for var_type in sample_res:
        if var_type not in inf_res:
            inf_res[var_type] = {}

        for var in sample_res[var_type]:
            if var not in inf_res[var_type]:
                inf_res[var_type][var] = {}

            sample_expanded = np.expand_dims(sample_res[var_type][var], axis=0)
            if len(inf_res[var_type][var]) == 0:
                # i.e. for first repeat (submodel)
                inf_res[var_type][var] = sample_expanded
            else:
                inf_res[var_type][var] = np.concatenate((inf_res[var_type][var], sample_expanded), axis=0)

    return inf_res


def add_sample_metrics_to_split_results(sample_metrics: dict,
                                        split_metrics_tmp: dict) -> dict:

    # for first sample in the dataloader
    is_first_sample = len(split_metrics_tmp) == 0

    for var_type in sample_metrics:
        if var_type not in split_metrics_tmp:
            split_metrics_tmp[var_type] = {}
        for var in sample_metrics[var_type]:
            sample_metric_expanded = np.expand_dims(sample_metrics[var_type][var], axis=0)
            if is_first_sample:
                split_metrics_tmp[var_type][var] = sample_metric_expanded
            else:
                split_metrics_tmp[var_type][var] = np.concatenate((split_metrics_tmp[var_type][var],
                                                                   sample_metric_expanded),
                                                                  axis=0)

    return split_metrics_tmp


def get_metadata_for_sample_metrics(metadata: dict) -> dict:

    # add the filename/path to the metadata so you can plot the metrics for each
    # sample and have an idea which samples are hard to segment, if there are some outliers in the data,
    # label noise, etc.
    try:
        dir_in, fname = os.path.split(metadata)
        sample_name, f_ext = os.path.splitext(fname)
        sample_metadata = {
            'metadata_filepath': np.expand_dims(np.array((metadata)), axis=0),
            'sample_name': np.expand_dims(np.array((sample_name)), axis=0)
        }
    except Exception as e:
        logger.warning('Problem getting the metadata? error = "{}",\n'
                      'This tested now only for MINIVESS that has the filepath in the metadata'.format(e))

    return sample_metadata


def merge_nested_dicts(a: dict, b: dict, path=[]):
    # https://stackoverflow.com/a/7205107
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_nested_dicts(a[key], b[key], path + [str(key)])
            elif a[key] != b[key]:
                raise Exception('Conflict at ' + '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def compute_split_metric_stats(split_metrics_tmp: dict,
                               var_types_to_keep: tuple = ('metrics', 'timing')) -> dict:

    split_stats = {}

    for var_type in split_metrics_tmp:
        if var_type in var_types_to_keep:  ## hard to compute stats for metadata for example
            split_stats[var_type] = {}
            for var in split_metrics_tmp[var_type]:
                split_stats[var_type][var] = {}
                input_data = split_metrics_tmp[var_type][var]
                split_stats[var_type][var] = compute_stats_of_array_in_dict(input_data)

    return split_stats


def compute_stats_of_array_in_dict(input_data: np.ndarray) -> dict:

    stats_out = {}
    stats_out['n'] = input_data.shape[0]
    stats_out['mean'] = np.mean(input_data)
    stats_out['var'] = np.var(input_data)

    return stats_out