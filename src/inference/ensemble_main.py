import os
from copy import deepcopy

import numpy as np
import torch
from loguru import logger

from tqdm import tqdm

from src.inference.ensemble_utils import add_sample_results_to_ensemble_results, add_sample_metrics_to_split_results, \
    compute_split_metric_stats
from src.inference.inference_utils import inference_sample, inference_best_repeat, \
    get_inference_metrics
from src.log_ML.model_saving import import_model_from_path
from src.utils.data_utils import redefine_dataloader_for_inference


def reinference_dataloaders(input_dict: dict,
                            dataloaders: dict,
                            artifacts_output_dir: str,
                            config: dict,
                            device,
                            model_scheme: str = 'ensemble_from_repeats',
                            debug_mode: bool = False):

    os.makedirs(artifacts_output_dir, exist_ok=True)
    results_out = {}
    for split in dataloaders:

        if split == 'TEST':
            # add this check to some debug mode to speed up development, and not having to compute all the splits

            logger.info('Inference for split "{}"'.format(split))
            if isinstance(dataloaders[split], dict):
                # i.e. this is either VAL or TEST. You could validate (save best model) based on multiple datasets if
                # desired at some point, and similarly you could have n external datasets that you would like to evaluate
                # and see how well you generalize for 3rd party data (out-of-distribution data, OOD)
                for dataset_name in dataloaders[split]:
                    logger.info('Dataset "{}"'.format(dataset_name))
                    dataloader = redefine_dataloader_for_inference(dataloader_batched=dataloaders[split][dataset_name],
                                                                   dataset_name=dataset_name,
                                                                   split=split,
                                                                   device=device,
                                                                   config=config)

                    if model_scheme == 'ensemble_from_repeats':
                        results_out[split] = inference_ensemble_dataloader(dataloader=dataloader,
                                                                           split=split,
                                                                           repeat_results=input_dict,
                                                                           config=config,
                                                                           device=device)
                    elif model_scheme == 'best_repeats':
                        results_out[split] = inference_best_repeat(dataloader=dataloader,
                                                                   split=split,
                                                                   best_repeat_dicts=input_dict,
                                                                   config=config,
                                                                   device=device)

                    else:
                        raise NotImplementedError('Unknown or not yet implemented '
                                                  'model_scheme = "{}"'.format(model_scheme))

            else:
                # TRAIN had no possibility to use multiple datasets (you could train for sure for multiple datasets,
                # but in the end this dataloader has to contain the samples from all those different datasets)
                dataset_name = 'MINIVESS'  # placeholder now as the train split comes with no dataset key, FIX later?
                dataloader = redefine_dataloader_for_inference(dataloader_batched=dataloaders[split],
                                                               dataset_name=dataset_name,
                                                               split=split,
                                                               device=device,
                                                               config=config)

                if model_scheme == 'ensemble_from_repeats':
                    results_out[split] = inference_ensemble_dataloader(dataloader=dataloader,
                                                                       split=split,
                                                                       repeat_results=input_dict,
                                                                       config=config,
                                                                       device=device)
                elif model_scheme == 'best_repeats':
                    results_out[split] = inference_best_repeat(dataloader=dataloader,
                                                               split=split,
                                                               best_repeat_dicts=input_dict,
                                                               config=config,
                                                               device=device)

                else:
                    raise NotImplementedError('Unknown or not yet implemented '
                                              'model_scheme = "{}"'.format(model_scheme))

    return results_out


def inference_ensemble_dataloader(dataloader,
                                  split: str,
                                  repeat_results: dict,
                                  config: dict,
                                  device: str):

    no_samples = len(dataloader.sampler)
    no_repeats = len(repeat_results)

    split_metrics = {}
    split_metrics_stat = {}

    # ASSUMING THAT all the repeats are the same (which should hold, and if you want to do diverse ensembles
    # later, keep the repeat and the architecture/model tweaks as separate)
    repeat_names = list(repeat_results.keys())
    repeat_result_example = repeat_results[repeat_names[0]]['best_dict']

    # DOUBLE-CHECK, why we have actually have dataset "twice", should this be removed?
    for d, dataset in enumerate(repeat_result_example):
        split_metrics[dataset] = {}
        split_metrics_stat[dataset] = {}

        for m, metric_to_track in enumerate(repeat_result_example[dataset]):
            split_metrics[dataset][metric_to_track] = {}
            split_metrics_stat[dataset][metric_to_track] = {}

            model_dict = repeat_result_example[dataset][metric_to_track]['model']
            model, _, _, _ = import_model_from_path(model_path=model_dict['model_path'],
                                                    validation_config=config['config']['VALIDATION'])

            # FIXME: get this from config
            metric_dict = {'roi_size': (64, 64, 8), 'sw_batch_size': 4, 'predictor': model, 'overlap': 0.6}
            model.eval()

            with (torch.no_grad()):
                for batch_idx, batch_data in enumerate(
                    tqdm(dataloader, desc='ENSEMBLE: Inference on dataloader samples, split "{}"'.format(split),
                         position=0)):
                    inference_results = {}

                    # tqdm(repeat_results, desc='Inference on repeat', position=1, leave=False)
                    for r, repeat_name in enumerate(repeat_results):

                        sample_res = inference_sample(batch_data,
                                                      model=model,
                                                      metric_dict=metric_dict,
                                                      device=device,
                                                      auto_mixedprec=config['config']['TRAINING']['AMP'])

                        # Add different repeats together so you can get the ensemble response
                        inference_results = add_sample_results_to_ensemble_results(inf_res=deepcopy(inference_results),
                                                                                   sample_res=sample_res)

                    # We have now the inference output of each repeat in the "ensemble_results" and we can for example
                    # get the average probabilities per pixel/voxel, or do majority voting
                    # This contains the binary mask that you could actually use
                    ensemble_stat_results = ensemble_repeats(inf_res=inference_results, config=config)

                    # And let's compute the metrics from the ensembled prediction
                    sample_ensemble_metrics = (
                        get_inference_metrics(ensemble_stat_results=ensemble_stat_results,
                                              y_pred=ensemble_stat_results['arrays']['mask'],
                                              config=config,
                                              batch_data=batch_data))

                    # Collect the metrics for each sample so you can for example compute mean dice for the split
                    split_metrics[dataset][metric_to_track] = \
                        add_sample_metrics_to_split_results(sample_metrics=deepcopy(sample_ensemble_metrics),
                                                            split_metrics_tmp=split_metrics[dataset][metric_to_track])

            # Done with the dataloader here and you have metrics computed per each of the sample
            # in the dataloader and you would like to probably have like mean Dice and stdev in Dice
            # along with the individual values so you could plot some distribution and highlight the
            # poorly performing samples
            split_metrics_stat[dataset][metric_to_track] = (
                compute_split_metric_stats(split_metrics_tmp=split_metrics[dataset][metric_to_track]))

    return {'samples': split_metrics, 'stats': split_metrics_stat}


def ensemble_repeats(inf_res: dict,
                     config: dict,
                     var_type_key: str = 'arrays',
                     var_key: str = 'probs') -> dict:

    input_data = inf_res[var_type_key][var_key]
    ensemble_stats = compute_ensembled_response(input_data, config)

    return ensemble_stats


def compute_ensembled_response(input_data: np.ndarray,
                               config: dict) -> dict:

    variable_stats = {}
    variable_stats['scalars'] = {}
    variable_stats['scalars']['n_samples'] = input_data.shape[0]  # i.e. number of repeats / submodels

    variable_stats['arrays'] = {}
    variable_stats['arrays']['mean'] = np.mean(input_data, axis = 0)  # i.e. number of repeats / submodels
    variable_stats['arrays']['var'] = np.var(input_data, axis=0)  # i.e. number of repeats / submodels
    variable_stats['arrays']['UQ_epistemic'] = np.nan
    variable_stats['arrays']['UQ_aleatoric'] = np.nan
    variable_stats['arrays']['entropy'] = np.nan

    ensemble_from_mean = True  # quick'dirty placeholder, add later to config the options
    mask_threshold = 0.5
    if ensemble_from_mean:
        variable_stats['arrays']['mask'] = (variable_stats['arrays']['mean'] > mask_threshold).astype('float32')
    else:
        a = 'majority_voting_here'

    return variable_stats


