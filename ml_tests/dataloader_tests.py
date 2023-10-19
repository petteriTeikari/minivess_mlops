import os
import random

import monai.data
import numpy as np
import torch

from loguru import logger

from ml_tests.dataset_tests import ml_test_dataset_for_allowed_types
from ml_tests.test_utils import add_boolean_and_metric_strings_to_summary
from src.inference.ensemble_utils import merge_nested_dicts


def ml_test_dataloader_dict_integrity(dataloaders: dict,
                                      test_config: dict,
                                      run_params: dict,
                                      debug_testing: bool = True):

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
                                                  test_config=test_config,
                                                  debug_testing=debug_testing))

            elif isinstance(split, monai.data.DataLoader):
                dataset = 'dummy'
                dataloader = dataloaders[fold_name][split_name]
                (dataloader_metrics[fold_name][split_name][dataset],
                 dataloader_ok_dict[fold_name][split_name][dataset]) = (
                    ml_test_single_dataloader(dataloader,
                                              split_name=split_name,
                                              fold_name=fold_name,
                                              test_config=test_config,
                                              debug_testing=debug_testing))
            else:
                raise IOError('Why is type of {} here?'.format(type(split)))

    # Create summary of the possibly problematic files in the dataloaders
    all_tests_ok, report_string = (
        dataloader_test_summary(dataloader_metrics=dataloader_metrics,
                                dataloader_ok_dict=dataloader_ok_dict))

    if not all_tests_ok:
        logger.debug('Here is your chance to write the report string to disk as well as .txt')
        print(report_string)


def dataloader_test_summary(dataloader_metrics: dict,
                            dataloader_ok_dict: dict,
                            params_to_print: tuple = ('number', 'filename', 'filepath')):

    report_string = 'HEADER PLACEHOLDER\n'
    all_tests_ok = True
    for fold_name in dataloader_metrics:
        report_string += f'{fold_name}\n'
        for split_name in dataloader_metrics[fold_name]:
            report_string += f'\t{split_name}\n'
            for dataset in dataloader_metrics[fold_name][split_name]:
                if dataset is not "dummy":
                    report_string += f'\t\t{dataset}\n'
                metrics = dataloader_metrics[fold_name][split_name][dataset]
                booleans = dataloader_ok_dict[fold_name][split_name][dataset]

                report_string, all_tests_ok = (
                    add_boolean_and_metric_strings_to_summary(metrics=metrics,
                                                              booleans=booleans,
                                                              report_string=report_string,
                                                              all_tests_ok=all_tests_ok,
                                                              params_to_print=params_to_print))

    if all_tests_ok:
        logger.info('All the dataloaders passed the tests!')
        report_string = None
    else:
        logger.error('Some of the dataloaders failed the tests!')
        raise OSError('Some of the dataloaders failed the tests!')

    return all_tests_ok, report_string


def ml_test_single_dataloader(dataloader: monai.data.DataLoader,
                              test_config: dict,
                              split_name: str = None,
                              fold_name: str = None,
                              dataset: str = None,
                              debug_testing: bool = False):

    dataloader_metrics, dataloader_ok_dict = {}, {}

    # Test placeholder to check that all the types are allowed, e.g. no None types allowed
    dataloader_ok_dict['dataset_has_valid_types'], dataloader_metrics['dataset_has_valid_types'] = (
        ml_test_dataloader_for_allowed_types(dataloader=dataloader,
                                             split_name=split_name,
                                             fold_name=fold_name,
                                             dataset=dataset))

    if dataloader_ok_dict['dataset_has_valid_types']:

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
                                          filenames=batch_data['metadata']['filepath_json'],
                                          no_batches=no_batches,
                                          debug_testing=debug_testing))

            # Collect batch-levels to a dataloader level results
            dataloader_metrics, dataloader_ok_dict = (
                dataloader_collector_of_batches(dataloader_metrics, dataloader_ok_dict,
                                                batch_metrics, batch_ok_dict,
                                                add_dummy_data=debug_testing))
    else:
        logger.warning('Cannot run nan/inf checks as you have illegal entries in the dataset(s) of your dataloader')

    return dataloader_metrics, dataloader_ok_dict


def dataloader_collector_of_batches(dataloader_metrics, dataloader_ok_dict,
                                    batch_metrics, batch_ok_dict,
                                    add_dummy_data: bool = False):

    if len(dataloader_metrics) == 0:
        dataloader_metrics = batch_metrics
    else:
        # This dictionary contains test-specific samples that had issues so that they can be
        # printed on the screen in the end if you need to hunt down issues with the input files
        if add_dummy_data:
            logger.warning("WARNING! Adding dummy data to the dataloader metrics")
            keys_so_far = list(batch_metrics['image_nans_infs'].keys())
            batch_metrics['image_nans_infs']['dummy{}'.format(random.randint(3, 22))] = (
                batch_metrics)['image_nans_infs'][keys_so_far[0]]
        dataloader_metrics = merge_nested_dicts(dataloader_metrics, batch_metrics)

    if len(dataloader_ok_dict) == 0:
        # if any batch of the dataloader is not ok, then the whole dataloader is not ok
        dataloader_ok_dict = batch_ok_dict
    else:
        for test_name in batch_ok_dict:
            if test_name not in dataloader_ok_dict:
                dataloader_ok_dict[test_name] = True
            previous_boolean = dataloader_ok_dict[test_name]
            current_boolean = batch_ok_dict[test_name]
            updated_boolean = previous_boolean and current_boolean
            dataloader_ok_dict[test_name] = updated_boolean

    # e.g. dataloader_ok_dict:
    #      {'image_nans_infs': False, 'label_nans_infs': False}
    # e.g. dataloader_metrics:
    #      {'image_nans_infs': {'mv16': {'number': 5, 'percentage': 0.1, 'filename': 'mv16.json'},
    #                           'mv52': {'number': 15, 'percentage': 1, 'filename': 'mv52.json'}},
    #       'label_nans_infs': ...
    #      }

    return dataloader_metrics, dataloader_ok_dict


def ml_test_single_batch_dict(batch_data: dict,
                              test_config: dict,
                              split_name: str,
                              fold_name: str,
                              dataset: str,
                              i: str,
                              filenames: list,
                              no_batches: int,
                              debug_testing: bool = False):

    batch_test_metrics, batch_tests = {}, {}

    test_name = 'image_nans_infs'
    batch_test_metrics[test_name], batch_tests[test_name] = (
        ml_test_batch_nan_and_inf(batch_tensor=batch_data['image'],
                                  test_name=test_name,
                                  filenames=filenames,
                                  fake_a_nan=debug_testing))

    test_name = 'label_nans_infs'
    batch_test_metrics[test_name], batch_tests[test_name] = (
        ml_test_batch_nan_and_inf(batch_tensor=batch_data['label'],
                                  test_name=test_name,
                                  filenames=filenames,
                                  fake_a_nan=debug_testing))

    return batch_test_metrics, batch_tests


def check_batch_type(batch_data: dict):

    if not isinstance(batch_data['image'], torch.Tensor):
        logger.error('Your batch data is not a torch.Tensor, but {}', type(batch_data['image']))
        logger.error('This could happen when using the vanilla Dataset when the paths to files are here')
        # Implement this if needed, but you might as well can use the CachedDataset with cache rate of 0.0
        raise IOError('Your batch data is not a torch.Tensor, but {}'.format(type(batch_data['image'])))


def ml_test_batch_nan_and_inf(batch_tensor: monai.data.MetaTensor,
                              test_name: str,
                              filenames: list,
                              fake_a_nan: bool = False):

    batch_test_metrics = {}
    assert len(batch_tensor.shape) == 5, ('Only 5D input tensors supported now'
                                          '(batch, channels, x, y, z)')

    batch_tensor = batch_tensor.detach().cpu().numpy()
    nan_true = np.isnan(batch_tensor)
    inf_true = np.isinf(batch_tensor)

    # samplewise sums
    nan_sums = np.sum(nan_true, axis=(1, 2, 3, 4))
    inf_sample_sums = np.sum(inf_true, axis=(1, 2, 3, 4))

    if fake_a_nan:
        logger.warning('WARNING! Faking a NaN in the batch tensor for testing purposes')
        nan_sums[0] = 5

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
                error_name = 'nan'
                if error_name not in batch_test_metrics:
                    batch_test_metrics = (
                        get_per_sample_naninf_stats(batch_test_metrics,
                                                    error_name = error_name,
                                                    error_sums = nan_sums,
                                                    sample_idx = sample_idx,
                                                    filepath = filepath,
                                                    fname = fname,
                                                    fname_wo_ext = fname_wo_ext,
                                                    no_voxels = no_voxels))


            if if_sample_has_infs[sample_idx]:
                error_name = 'inf'
                batch_test_metrics = (
                    get_per_sample_naninf_stats(batch_test_metrics,
                                                error_name=error_name,
                                                error_sums=inf_sample_sums,
                                                sample_idx=sample_idx,
                                                filepath=filepath,
                                                fname=fname,
                                                fname_wo_ext=fname_wo_ext,
                                                no_voxels=no_voxels))

    return batch_test_metrics, batch_test_ok


def get_per_sample_naninf_stats(batch_test_metrics: dict,
                                error_name: str,
                                error_sums: np.ndarray,
                                sample_idx: int,
                                filepath: str,
                                fname: str,
                                fname_wo_ext: str,
                                no_voxels: int) -> dict:

    batch_test_metrics[fname_wo_ext] = {}
    batch_test_metrics[fname_wo_ext]['number'] = error_sums[sample_idx]
    batch_test_metrics[fname_wo_ext]['percentage'] = (
            100*(batch_test_metrics[fname_wo_ext]['number'] / no_voxels))
    batch_test_metrics[fname_wo_ext]['filename'] = fname
    batch_test_metrics[fname_wo_ext]['filepath'] = filepath

    return batch_test_metrics


def ml_test_dataloader_for_allowed_types(dataloader: monai.data.dataloader.DataLoader,
                                         split_name: str,
                                         fold_name: str,
                                         dataset: str):
    """
    Test for possible Nones or any other illegal types for Pytorch dataloaders
    """

    dataloader_is_valid = True
    samples_not_valid = {}
    try:
        for i, batch_data in enumerate(dataloader):
            pass
    except Exception as e:
        dataloader_is_valid = False
        logger.error('Problem iterating through the dataloader, e= {}\n'.format(e))
        try:
            is_dataset_valid, samples_not_valid = (
                ml_test_dataset_for_allowed_types(split_file_dict=dataloader.dataset.data))
            assert dataloader_is_valid == is_dataset_valid, 'these should be the same!'
        except Exception as e:
            raise IOError('Failed to get the "split_file_dict" Some new type of dataloader? e = {}'.format(e))

    return dataloader_is_valid, samples_not_valid