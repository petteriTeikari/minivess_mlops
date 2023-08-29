import os
from copy import deepcopy

import numpy as np
import torch
from loguru import logger


def save_models_if_improved(best_dicts, epoch,
                            model, optimizer, lr_scheduler,
                            train_epoch_results, train_results,
                            eval_epoch_results, eval_results,
                            validation_config, config, model_dir,
                            split_key: str = 'VAL',
                            results_type: str = 'scalars'):

    os.makedirs(model_dir, exist_ok=True)
    best_dicts_out = {}

    if epoch >= validation_config['NO_WARMUP_EPOCHS']:
        for dataset in eval_epoch_results['VAL'].keys():
            best_dicts_out[dataset] = {}
            for m_idx, metric in enumerate(validation_config['METRICS_TO_TRACK']):
                best_dicts_out[dataset][metric] = {}

                # i.e. the first epoch that we are tracking so obviously the model is now considered having
                if best_dicts is None:

                    best_dicts_out[dataset][metric] = \
                        get_best_dict_from_current_epoch_results(eval_epoch_results, train_epoch_results)

                    current_value = get_current_metric_value(eval_epoch_results, eval_results,
                                                             validation_config, dataset, metric, epoch,
                                                             split_key=split_key, results_type=results_type)

                    # Run the model saving script (and any other stuff that you might want to do)
                    best_dicts_out[dataset][metric]['model'] = (
                        model_improved_script(best_dict=best_dicts_out[dataset][metric],
                                              current_value=current_value,
                                              best_value_so_far=np.nan,
                                              model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                              epoch=epoch, model_dir=model_dir,
                                              dataset=dataset, metric=metric,
                                              validation_config=validation_config))

                # After 1st epoch, when you have something on your best dict
                else:

                    best_dict = best_dicts[dataset][metric]
                    model_improved, current_value, best_value_so_far = \
                        check_if_value_improved(eval_epoch_results, eval_results,
                                                best_dict = best_dict,
                                                validation_config=validation_config, dataset=dataset,
                                                metric=metric, epoch=epoch,
                                                split_key=split_key, results_type=results_type,
                                                best_op=validation_config['METRICS_TO_TRACK_OPERATORS'][m_idx])

                    if model_improved:

                        # Update the best dict from the current epoch
                        best_dicts_out[dataset][metric] = \
                            get_best_dict_from_current_epoch_results(eval_epoch_results, train_epoch_results)

                        # Run the model saving script (and any other stuff that you might want to do)
                        best_dicts_out[dataset][metric]['model'] = (
                            model_improved_script(best_dict=best_dict,
                                                  current_value=current_value, best_value_so_far=best_value_so_far,
                                                  model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                                  epoch=epoch, model_dir=model_dir,
                                                  dataset=dataset, metric=metric,
                                                  validation_config=validation_config))

                    else:

                        best_dicts_out[dataset][metric] = deepcopy(best_dicts[dataset][metric])

    else:
        logger.debug('epoch {} not past warmup epochs ({}) '
                     'yet -> not tracking model improvement yet'.
                     format(epoch, validation_config['NO_WARMUP_EPOCHS']))

    return best_dicts_out


def check_if_value_improved(eval_epoch_results: dict, eval_results: dict, best_dict: dict,
                             validation_config: dict, dataset: str, metric: str, epoch: int,
                             split_key = 'VAL', results_type = 'scalars', best_op:str = 'max'):

    assert len(validation_config['METRICS_TO_TRACK']) == len(validation_config['METRICS_TO_TRACK_OPERATORS']), \
        'You should have as many metrics (e.g. Dice, Hausdorff Distance) ' \
        'as you have the "operators" indicating what is better (e.g. "max", "min")\n' \
        'See contents of "validation_config":\n{}'.format(validation_config)

    current_value = get_current_metric_value(eval_epoch_results, eval_results,
                                             validation_config, dataset, metric, epoch,
                                             split_key=split_key, results_type=results_type)

    best_value_so_far = best_dict['eval_epoch_results'][split_key][dataset][results_type][metric]

    model_improved = False
    if best_op == 'max':
        if current_value > best_value_so_far:
            model_improved = True

    elif best_op == 'min':
        if current_value < best_value_so_far:
            model_improved = True

    return model_improved, current_value, best_value_so_far


def get_current_metric_value(eval_epoch_results: dict, eval_results: dict,
                             validation_config: dict, dataset: str, metric: str, epoch: int,
                             split_key = 'VAL', results_type = 'scalars') -> float:

    if validation_config['EMA_SMOOTH']:
        # smooth the history to avoid spurious noisy metrics, and taking the value of current epoch
        raise NotImplementedError('Implement Exponential Moving Average')

    else:
        value = eval_epoch_results[split_key][dataset][results_type][metric]

    return value


def model_improved_script(best_dict: dict,
                          current_value: float, best_value_so_far: float,
                          model, optimizer, lr_scheduler,
                          epoch: int, model_dir: str,
                          dataset: str, metric: str, validation_config: dict) -> None:
    """
    Instead of "Github demo repos" in which you only want to save the weights, we also want to save all possible
    values (that you track in the results) associated with the best model. Like what was some other metric when
    your Dice was the best for this model, what figure you had saved for this best model, what dataframe containing
    some tabular data had for this best model, etc.

    :param best_dict:
    :param current_value:
    :param best_value_so_far:
    :param model:
    :param optimizer:
    :param lr_scheduler:
    :param epoch:
    :param output_dir:
    :param dataset:
    :param metric:
    :param validation_config:
    :return:
    """

    model_fname_base = "bestModel__{}__{}".format(dataset, metric)
    if validation_config['SAVE_FULL_MODEL']:
        checkpoint = dict(
            epoch=epoch+1,
            model=model,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            lr_scheduler=lr_scheduler,
            best_dict=best_dict
        )
    else:
        checkpoint = dict(
            epoch=epoch + 1,
            model=None,
            state_dict=model.state_dict(),
            optimizer=optimizer.state_dict(),
            lr_scheduler=lr_scheduler,
            best_dict=best_dict
        )

    path_out = os.path.join(model_dir, model_fname_base + '.pth')
    torch.save(checkpoint, path_out)
    try:
        filesize = (os.stat(path_out).st_size) / (1024 * 1024)
    except Exception as e:
        # if happens, not a big deal, you are not just displaying the size correctly
        logger.warning('Problem getting model file size from disk to display, e = {}'.format(e))
        filesize = np.nan

    logger.debug('"{}" improved from {:.4f} to {:.4f} (dataset = {}) '
                 '-> saving this to disk (model size = {:.1f} MB)'.
                 format(metric, best_value_so_far, current_value, dataset,filesize))

    return {'model_path': path_out}


def import_model_from_path(model_path: str,
                           validation_config: dict):

    if os.path.exists(model_path):
        if validation_config['SAVE_FULL_MODEL']:
            model_dict = torch.load(model_path)
            model = model_dict['model']
            model.load_state_dict(model_dict['state_dict'])
            # optimizer, lr_scheduler also saved if you want to use this function later for resuming/finetuning
            optimizer = model_dict['optimizer']
            lr_scheduler = model_dict['lr_scheduler']
            best_dict = model_dict['best_dict']
        else:
            raise NotImplementedError('Inference implemented only now for full model saving,\n'
                                      'not for "just weights" saving')

    else:
        raise IOError('The model file does not exist on disk? (path = {})\n'
                      'This should not really happen as we managed to save these during training'.format(model_path))

    return model, best_dict, optimizer, lr_scheduler


def get_best_dict_from_current_epoch_results(eval_epoch_results: dict, train_epoch_results: dict):

    # TOCHECK not totally sure if the deepcopy() is needed, but previously had this mystery glitch
    # of dictionaries getting weirdly updated
    result_dicts = {
                    'eval_epoch_results': deepcopy(eval_epoch_results),
                    'train_epoch_results': deepcopy(train_epoch_results)
                    }

    return result_dicts


