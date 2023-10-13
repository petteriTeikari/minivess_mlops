import os

import monai.data
import numpy as np

from loguru import logger


def ml_test_dataloader_dict_integrity(dataloaders: dict,
                                      test_config: dict):

    dataloader_metrics, dataloader_ok_dict = {}, {}
    # Loop through all the dataloaders
    # TODO! You need to modify this later also to include the dataloader
    #  per architecture
    for fold_name in dataloaders:
        dataloader_metrics[fold_name], dataloader_ok_dict[fold_name] = {}, {}
        fold = dataloaders[fold_name]
        for split_name in fold:
            dataloader_metrics[fold_name][split_name], dataloader_ok_dict[fold_name][split_name] = {}, {}
            split = fold[split_name]
            if isinstance(split, dict):
                for dataset in dataloaders[fold_name][split_name]:
                    dataloader = dataloaders[fold_name][split_name][dataset]
                    (dataloader_metrics[fold_name][split_name][dataset],
                     dataloader_ok_dict[fold_name][split_name][dataset]) = (
                        ml_test_single_dataloader(dataloader,
                                                  split_name=split_name,
                                                  fold_name=fold_name,
                                                  dataset=dataset,
                                                  test_config=test_config))

            elif isinstance(split, monai.data.DataLoader):
                dataset = 'train'
                dataloader = dataloaders[fold_name][split_name]
                (dataloader_metrics[fold_name][split_name][dataset],
                 dataloader_ok_dict[fold_name][split_name][dataset]) = (
                    ml_test_single_dataloader(dataloader,
                                              split_name=split_name,
                                              fold_name=fold_name,
                                              test_config=test_config))
            else:
                raise IOError('Why is type of {} here?'.format(type(split)))

    # TODO! Now analyze all the dataloaders for possible errors
    #  We did not want to raise an error immediately when an error is found as you most likely have
    #  multiple errors in your dataset and you would like to maybe have a list of all possible errors
    #  if your preprocessing operator is doing something funky for specific files, or whatever?
    dataloaders_ok, dataloaders_metrics = (
        dataloader_test_summary(dataloader_metrics=dataloader_metrics,
                                dataloader_ok_dict=dataloader_ok_dict))

    return dataloaders_ok, dataloaders_metrics


def dataloader_test_summary(dataloader_metrics: dict,
                            dataloader_ok_dict: dict):

    return {}, True


def ml_test_single_dataloader(dataloader: monai.data.DataLoader,
                              test_config: dict,
                              split_name: str = None,
                              fold_name: str = None,
                              dataset: str = None):

    # Check that the dataloader does not have dictionaries that for example
    # contain "None" or something that will crash your training script
    ml_test_dataloader_for_allowed_types(dataloader=dataloader)

    dataloader_metrics, dataloader_ok_dict = {}, {}
    no_batches = len(dataloader)
    for i, batch_data in enumerate(dataloader):
        logger.info('ML Test for batch {}/{} in split = {}, fold = {}, dataset = {}'.
                    format(i+1, no_batches, split_name, fold_name, dataset))

        batch_metrics, batch_ok_dict = (
            ml_test_single_batch_dict(batch_data,
                                      test_config=test_config,
                                      split_name=split_name,
                                      fold_name=fold_name,
                                      dataset=dataset,
                                      i=i,
                                      filenames=batch_data['metadata'],
                                      no_batches=no_batches))

        # TODO! combine the dictionaries here so that all the batch dicts are collected
        #  to one dataloader dict
        dataloader_metrics, dataloader_ok_dict = batch_metrics, batch_ok_dict

    return dataloader_metrics, dataloader_ok_dict


def ml_test_single_batch_dict(batch_data: dict,
                              test_config: dict,
                              split_name: str,
                              fold_name: str,
                              dataset: str,
                              i: str,
                              filenames: list,
                              no_batches: int):

    batch_test_metrics, batch_tests = {}, {}

    test_name = 'image_nans_infs'
    batch_test_metrics[test_name], batch_tests[test_name] = (
        ml_test_batch_nan_and_inf(batch_tensor=batch_data['image'],
                                  test_name=test_name,
                                  filenames=filenames))

    test_name = 'label_nans_infs'
    batch_test_metrics[test_name], batch_tests[test_name] = (
        ml_test_batch_nan_and_inf(batch_tensor=batch_data['label'],
                                  test_name=test_name,
                                  filenames=filenames))

    return batch_test_metrics, batch_tests


def ml_test_batch_nan_and_inf(batch_tensor: monai.data.MetaTensor,
                              test_name: str,
                              filenames: list):

    batch_test_metrics = {}
    assert len(batch_tensor.shape) == 5, ('Only 5D input tensors supported now'
                                          '(batch, channels, x, y, z)')

    batch_tensor = batch_tensor.detach().cpu().numpy()
    nan_true = np.isnan(batch_tensor)
    inf_true = np.isinf(batch_tensor)

    # samplewise sums
    nan_sums = np.sum(nan_true, axis=(1, 2, 3, 4))
    inf_sample_sums = np.sum(inf_true, axis=(1, 2, 3, 4))

    if_sample_has_nans = nan_sums > 0
    if_any_nan = np.sum(if_sample_has_nans) > 0
    if_sample_has_infs = inf_sample_sums > 0
    if_any_inf = np.sum(if_sample_has_infs) > 0

    batch_test_ok = True
    if if_any_nan or if_any_inf:
        batch_test_ok = False
        for sample_idx in range(len(if_sample_has_nans)):
            filepath = filenames[sample_idx]
            fname = os.path.split(filepath)[1]
            fname_wo_ext, ext = os.path.splitext(fname)
            no_voxels = np.size(np.squeeze(batch_tensor[sample_idx, :, :, :, :]))

            if if_sample_has_nans[sample_idx]:
                if 'nan' not in batch_test_metrics:
                    batch_test_metrics['nan'] = {}
                batch_test_metrics['nan']['number'] = nan_sums[sample_idx]
                batch_test_metrics['nan']['percentage'] = batch_test_metrics['nan']['number'] / no_voxels
                batch_test_metrics['nan']['filename'] = fname
                batch_test_metrics['nan']['filepath'] = filepath

            if if_sample_has_infs[sample_idx]:
                if 'inf' not in batch_test_metrics:
                    batch_test_metrics['inf'] = {}
                batch_test_metrics['inf']['number'] = nan_sums[sample_idx]
                batch_test_metrics['inf']['percentage'] = batch_test_metrics['nan']['number'] / no_voxels
                batch_test_metrics['inf']['filename'] = fname
                batch_test_metrics['inf']['filepath'] = filepath

    return batch_test_metrics, batch_test_ok


def ml_test_dataloader_for_allowed_types(dataloader):
    # TODO! For example you cannot have None types in any metadata that goes through
    #  the dataloader and simply by iterating stuff through here will throw errors on
    #  the glitch sample
    pass