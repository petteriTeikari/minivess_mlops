import os
from loguru import logger
from omegaconf import DictConfig

from src.inference.ensemble_model import inference_ensemble_dataloader
from src.inference.ensemble_utils import get_ensemble_name, get_submodel_name
from src.inference.inference_utils import inference_best_repeat
from src.utils.dataloader_utils import redefine_dataloader_for_inference
from src.utils.dict_utils import cfg_key


def reinference_dataloaders(input_dict: dict,
                            dataloaders: dict,
                            artifacts_output_dir: str,
                            cfg: dict,
                            device,
                            model_scheme: str = 'ensemble_from_repeats',
                            debug_mode: bool = False):

    # TODO! add from debug_mode, the done splits, as in the "debug" mode, we are now doing the TEST only
    allowed_splits = ['TEST']

    os.makedirs(artifacts_output_dir, exist_ok=True)
    results_out = {}
    for split in dataloaders:

        if split in allowed_splits:
            # add this check to some debug mode to speed up development, and not having to compute all the splits

            logger.info('Inference for split "{}"'.format(split))
            if isinstance(dataloaders[split], dict):
                # i.e. this is either VAL or TEST. You could validate (save best model) based on multiple datasets if
                # desired at some point, similarly you could have n external datasets that you would like to evaluate
                # and see how well you generalize for 3rd party data (out-of-distribution data, OOD)
                for dataset_name in dataloaders[split]:
                    logger.info('Dataset "{}"'.format(dataset_name))
                    dataloader_batched = cfg_key(dataloaders, split, dataset_name)
                    dataloader = redefine_dataloader_for_inference(dataloader_batched=dataloader_batched,
                                                                   dataset_name=dataset_name,
                                                                   split=split,
                                                                   device=device,
                                                                   cfg=cfg)

                    if model_scheme == 'ensemble_from_repeats':
                        results_out[split] = inference_ensembles_dataloader(dataloader=dataloader,
                                                                            split=split,
                                                                            archi_results=input_dict,
                                                                            cfg=cfg,
                                                                            device=device)
                    elif model_scheme == 'best_repeats':
                        results_out[split] = inference_best_repeat(dataloader=dataloader,
                                                                   split=split,
                                                                   best_repeat_dicts=input_dict,
                                                                   cfg=cfg,
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
                                                               cfg=cfg)
                if model_scheme == 'ensemble_from_repeats':
                    results_out[split] = inference_ensembles_dataloader(dataloader=dataloader,
                                                                        split=split,
                                                                        archi_results=input_dict,
                                                                        cfg=cfg,
                                                                        device=device)
                elif model_scheme == 'best_repeats':
                    results_out[split] = inference_best_repeat(dataloader=dataloader,
                                                               split=split,
                                                               best_repeat_dicts=input_dict,
                                                               cfg=cfg,
                                                               device=device)
                else:
                    raise NotImplementedError('Unknown or not yet implemented '
                                              'model_scheme = "{}"'.format(model_scheme))

        else:
            logger.debug('Skipping split = {}'.format(split))

    return results_out


def inference_ensembles_dataloader(dataloader,
                                   split: str,
                                   archi_results: dict,
                                   cfg: dict,
                                   device: str):

    # ASSUMING THAT all the repeats are the same (which should hold, and if you want to do diverse ensembles
    # later, keep the repeat and the architecture/model tweaks as separate)
    architecture_names = list(archi_results.keys())
    first_arch_name = architecture_names[0]
    repeat_names = list(archi_results[first_arch_name].keys())
    first_repeat = archi_results[first_arch_name][repeat_names[0]]
    repeat_result_example = first_repeat['best_dict']

    no_submodels = len(architecture_names) * len(repeat_names)
    no_eval_datasets = len(repeat_result_example)
    no_tracked_metrics = len(repeat_result_example[list(repeat_result_example)[0]])
    no_ensembles = no_eval_datasets*no_tracked_metrics
    logger.info('No of submodels in ensemble = {} ({} repeats, {} architectures)'.
                format(no_submodels, len(repeat_names), len(architecture_names)))
    logger.info('No of ensembles = {} ({} eval_datasets, {} tracked metrics)'.
                format(no_ensembles, no_eval_datasets, no_tracked_metrics))

    _, ensemble_models_flat = collect_submodels_of_the_ensemble_archi(archi_results)

    ensemble_results = {}
    for i, ensemble_name in enumerate(ensemble_models_flat):
        logger.info('Inference on ensemble_name = "{}" (#{}/{})'.
                    format(ensemble_name, i+1, len(ensemble_models_flat)))
        ensemble_results[ensemble_name] = (
            inference_ensemble_dataloader(models_of_ensemble=ensemble_models_flat[ensemble_name],
                                          cfg=cfg,
                                          split=split,
                                          dataloader=dataloader,
                                          device=device))

    return ensemble_results


def collect_submodels_of_the_ensemble_archi(archi_results: dict):
    """
    Combine later with the collect_submodels_of_the_ensemble() that is basically the same
    just with one more nesting level
    """

    # Two ways of organizing the same saved models, deprecate the other later
    model_paths = {}  # original nesting notation
    ensemble_models_flat = {}  # more intuitive maybe, grouping models under the same ensemble name
    n_submodels = 0
    n_ensembles = 0

    for archi_name in archi_results:
        model_paths[archi_name] = {}
        for repeat_name in archi_results[archi_name]:
            model_paths[archi_name][repeat_name] = {}
            best_dict = archi_results[archi_name][repeat_name]['best_dict']
            for ds in best_dict:
                model_paths[archi_name][repeat_name][ds] = {}
                for tracked_metric in best_dict[ds]:

                    model_path = best_dict[ds][tracked_metric]['model']['model_path']
                    model_paths[archi_name][repeat_name][ds][tracked_metric] = model_path

                    ensemble_name = get_ensemble_name(dataset_validated=ds,
                                                      metric_to_track=tracked_metric)

                    if ensemble_name not in ensemble_models_flat:
                        ensemble_models_flat[ensemble_name] = {}
                        n_ensembles += 1

                    submodel_name = get_submodel_name(archi_name=archi_name,
                                                      repeat_name=repeat_name)

                    if submodel_name not in ensemble_models_flat[ensemble_name]:
                        ensemble_models_flat[ensemble_name][submodel_name] = model_path
                        n_submodels += 1

    # Remember that you get more distinct ensembles if use more tracking metrics (e.g. track for best loss and for
    # best Hausdorff distance), and if you validate for more subsets of the data instead of just having one
    # "vanilla" validation splöit
    logger.info('Collected a total of {} models, and {} distinct ensembles for ensemble inference'.
                format(n_submodels, n_ensembles))

    return model_paths, ensemble_models_flat
