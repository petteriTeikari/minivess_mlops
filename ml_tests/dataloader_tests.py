import os
import random

import monai.data
import numpy as np

from loguru import logger

from src.inference.ensemble_utils import merge_nested_dicts


def ml_test_dataloader_dict_integrity(dataloaders: dict,
                                      test_config: dict,
                                      run_params: dict,
                                      debug_the_testing: bool = True):

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
                                                  debug_the_testing=debug_the_testing))

            elif isinstance(split, monai.data.DataLoader):
                dataset = 'dummy'
                dataloader = dataloaders[fold_name][split_name]
                (dataloader_metrics[fold_name][split_name][dataset],
                 dataloader_ok_dict[fold_name][split_name][dataset]) = (
                    ml_test_single_dataloader(dataloader,
                                              split_name=split_name,
                                              fold_name=fold_name,
                                              test_config=test_config,
                                              debug_the_testing=debug_the_testing))
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

                for test_name in metrics:
                    test_boolean = booleans[test_name]
                    report_string += f'\t\t\t{test_name}: {test_boolean}\n'
                    test_metrics = metrics[test_name]
                    for problem_sample in test_metrics:
                        problem_dict = test_metrics[problem_sample]
                        report_string += f'\t\t\t\t{problem_sample}\n'
                        for key in params_to_print:
                            try:
                                report_string += f'\t\t\t\t\t{key} = {problem_dict[key]}\n'
                            except:
                                logger.warning('cannot find desired key = "{}" in the problem_dict'.format(key))
                    all_tests_ok = all_tests_ok and test_boolean

    if all_tests_ok:
        logger.info('All the dataloaders passed the tests!')
        report_string = None
    else:
        logger.error('Some of the dataloaders failed the tests!')

    return all_tests_ok, report_string


def ml_test_single_dataloader(dataloader: monai.data.DataLoader,
                              test_config: dict,
                              split_name: str = None,
                              fold_name: str = None,
                              dataset: str = None,
                              debug_the_testing: bool = False):

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
                                      no_batches=no_batches,
                                      debug_the_testing=debug_the_testing))

        # Collect batch-levels to a dataloader level results
        dataloader_metrics, dataloader_ok_dict = (
            dataloader_collector_of_batches(dataloader_metrics, dataloader_ok_dict,
                                            batch_metrics, batch_ok_dict,
                                            add_dummy_data=debug_the_testing))

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
        for test_name in dataloader_ok_dict:
            previous_boolean = dataloader_ok_dict[test_name]
            current_boolean = batch_ok_dict[test_name]
            dataloader_ok_dict[test_name] = previous_boolean and current_boolean

    return dataloader_metrics, dataloader_ok_dict


def ml_test_single_batch_dict(batch_data: dict,
                              test_config: dict,
                              split_name: str,
                              fold_name: str,
                              dataset: str,
                              i: str,
                              filenames: list,
                              no_batches: int,
                              debug_the_testing: bool = False):

    batch_test_metrics, batch_tests = {}, {}

    test_name = 'image_nans_infs'
    batch_test_metrics[test_name], batch_tests[test_name] = (
        ml_test_batch_nan_and_inf(batch_tensor=batch_data['image'],
                                  test_name=test_name,
                                  filenames=filenames,
                                  fake_a_nan=debug_the_testing))

    test_name = 'label_nans_infs'
    batch_test_metrics[test_name], batch_tests[test_name] = (
        ml_test_batch_nan_and_inf(batch_tensor=batch_data['label'],
                                  test_name=test_name,
                                  filenames=filenames,
                                  fake_a_nan=debug_the_testing))

    return batch_test_metrics, batch_tests


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
            batch_test_metrics[fname_wo_ext]['number'] / no_voxels)
    batch_test_metrics[fname_wo_ext]['filename'] = fname
    batch_test_metrics[fname_wo_ext]['filepath'] = filepath

    return batch_test_metrics


def ml_test_dataloader_for_allowed_types(dataloader):
    # TODO! For example you cannot have None types in any metadata that goes through
    #  the dataloader and simply by iterating stuff through here will throw errors on
    #  the glitch sample
    pass