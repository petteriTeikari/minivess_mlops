import json
import os
import tempfile
import warnings
from functools import singledispatch

import omegaconf
from omegaconf import OmegaConf
from loguru import logger

import numpy as np
import wandb.sdk.wandb_run
import yaml



def convert_value_to_numpy_array(value_in):

    if type(value_in) is np.ndarray:
        # if is "arrays", and already a Numpy array
        raise NotImplementedError('Implement when input is already np.ndarray!')

    elif isinstance(value_in, (float, int, np.floating, np.integer)):
        value_in = np.array(value_in)[np.newaxis]

    else:
        # e.g. if you save strings or something else here that is hard really to use for stats computations
        logger.warning('Unknown value type = {}'.format(type(value_in)))

    return value_in


def compute_numpy_stats(value_array_in: np.ndarray):

    stats_dict = {}
    no_of_folds, no_of_repeats = value_array_in.shape

    stats_dict['mean'] = np.mean(value_array_in)
    stats_dict['stdev'] = np.std(value_array_in)
    stats_dict['n'] = value_array_in.size

    return stats_dict


def get_number_of_steps_from_repeat_results(results: dict, result_type: str = 'train_results'):

    try:
        res = results[result_type]
        for split in res:
            for dataset in res[split]:

                if result_type == 'train_results':
                    sub_res = res[split][dataset]['arrays']
                elif result_type == 'eval_results':
                    sub_res = res[split][dataset]['scalars']
                else:
                    raise IOError('Unknown result_type = {}'.format(result_type))

                if len(sub_res) > 0:
                    first_array_as_ex = sub_res[list(sub_res.keys())[0]]
                    if isinstance(first_array_as_ex, float):
                        # special edge case when you only trained for one epoch
                        no_steps = 1
                    else:
                        no_steps = len(first_array_as_ex)
                else:
                    # if you have no metrics saved to "arrays", cannot get the metric
                    no_steps = np.nan

    except Exception as e:
        logger.warning('Problem getting number of steps ("{}"), e = {}'.format(result_type, e))
        no_steps = np.nan

    return no_steps


def write_config_as_yaml(config: dict, dir_out: str):

    path_out = os.path.join(dir_out, 'config.yaml')
    if isinstance(config, omegaconf.dictconfig.DictConfig):
        logger.info('Dumping the OmegaConf config to disk as .yaml ({})'.format(path_out))
        with tempfile.TemporaryDirectory() as d:
            OmegaConf.save(config, path_out)
            # test that this is actually the same
            with open(path_out) as f:
                x = yaml.unsafe_load(f)
                assert config == x, ('The OmegaConf dictionary dumped to disk as .yaml is not the sane as used '
                                     'for training, something funky happened during saving')
    else:
        logger.info('Dumping vanilla Python dictionary to disk as .yaml ({})'.format(path_out))
        try:
            with open(path_out, 'w') as outfile:
                yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)
        except Exception as e:
            logger.warning('Problem writing the config to disk, e = {}'.format(e))

    return path_out