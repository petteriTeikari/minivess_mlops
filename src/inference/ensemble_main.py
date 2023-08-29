import torch
from loguru import logger
from monai.inferers import sliding_window_inference


from src.log_ML.model_saving import import_model_from_path
from src.utils.data_utils import redefine_dataloader_for_inference


def ensemble_the_repeats(repeat_results: dict,
                         dataloaders,
                         config: dict,
                         device: str):

    for split in dataloaders:

        logger.info('Ensemble inference for split "{}"'.format(split))
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
                inference_ensemble_dataloader(dataloader=dataloader,
                                              repeat_results=repeat_results,
                                              config=config,
                                              device=device)

        else:
            # TRAIN had no possibility to use multiple datasets (you could train for sure for multiple datasets,
            # but in the end this dataloader has to contain the samples from all those different datasets)
            dataset_name = 'MINIVESS'  # placeholder now as the train split comes with no dataset key, FIX later?
            dataloader = redefine_dataloader_for_inference(dataloader_batched=dataloaders[split],
                                                           dataset_name=dataset_name,
                                                           split=split,
                                                           device=device,
                                                           config=config)
            inference_ensemble_dataloader(dataloader=dataloader,
                                          repeat_results=repeat_results,
                                          config=config,
                                          device=device)


def inference_ensemble_dataloader(dataloader,
                                  repeat_results: dict,
                                  config: dict,
                                  device: str):

    no_samples = len(dataloader.sampler)
    no_repeats = len(repeat_results)
    ensemble_results = {}

    for r, repeat_name in enumerate(repeat_results):

        repeat_inference_result = inference_repeat_dataloader(dataloader=dataloader,
                                                              repeat_result=repeat_results[repeat_name]['best_dict'],
                                                              config=config,
                                                              device=device)


def inference_repeat_dataloader(dataloader,
                                repeat_result: dict,
                                config: dict,
                                device: str):

    repeat_inference_result = {}

    # DOUBLE-CHECK, why we have actually have dataset "twice", should this be removed?
    for d, dataset in enumerate(repeat_result):
        repeat_inference_result[dataset] = {}
        for m, metric_to_track in enumerate(repeat_result[dataset]):
            model_dict = repeat_result[dataset][metric_to_track]['model']
            model, _, _, _ = import_model_from_path(model_path=model_dict['model_path'],
                                                    validation_config=config['config']['VALIDATION'])
            repeat_inference_result[dataset][metric_to_track] = (
                inference_model_dataloader(model=model,
                                           dataloader=dataloader,
                                           config=config,
                                           device=device,
                                           auto_mixedprec=config['config']['TRAINING']['AMP']))


def inference_model_dataloader(model,
                               dataloader,
                               config: dict,
                               device: str,
                               auto_mixedprec: bool = True):
    """
    See e.g. https://docs.monai.io/en/stable/inferers.html
             https://github.com/davidiommi/Ensemble-Segmentation/blob/main/predict_single_image.py
             https://github.com/Project-MONAI/tutorials/blob/main/modules/cross_validation_models_ensemble.ipynb
    :param model:
    :param dataloader:
    :param config:
    :param device:
    :param auto_mixedprec:
    :return:
    """

    # FIXME: get this from config
    metric_dict = {'roi_size': (64, 64, 8), 'sw_batch_size': 4, 'predictor': model, 'overlap': 0.6}

    model.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if auto_mixedprec:  ## AMP
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(inputs=batch_data["image"].to(device), **metric_dict)
            else:
                val_outputs = sliding_window_inference(inputs=batch_data["image"].to(device), **metric_dict)