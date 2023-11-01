import os

import numpy as np
import torch
from loguru import logger

from src.inference.ensemble_utils import merge_nested_dicts


def ml_test_dataset_for_allowed_types(split_file_dict: dict,
                                      exts: tuple = ('.nii.gz', '.nii')):

    samples_not_valid = {}
    no_samples = len(split_file_dict)
    is_dataset_valid = True
    for i, sample in enumerate(split_file_dict):
        # assuming now that 'image' and 'label' are okay, and as 'metadata' can be more free text,
        # it is possible that illegal entries get there
        fname = os.path.split(sample['image'])[1]
        sample_name = fname
        for ext in exts:
            sample_name = sample_name.replace(ext, '')

        metadata_dict = sample.get('metadata', {})
        problematic_keys = ml_test_metadata_dict(metadata_dict=metadata_dict, sample_name=sample_name)
        samples_not_valid = merge_nested_dicts(samples_not_valid, problematic_keys)

    if len(samples_not_valid) > 0:
        is_dataset_valid = False
        logger.warning('Dataset contains illegal types!')

    return is_dataset_valid, samples_not_valid


def ml_test_metadata_dict(metadata_dict: dict,
                          sample_name: str):

    problematic_keys = {}
    metadata_keys = list(metadata_dict.keys())
    for key in metadata_keys:
        if isinstance(metadata_dict[key], dict):
            ml_test_metadata_dict(metadata_dict[key], sample_name)
        else:
            value = metadata_dict[key]
            is_valid_type = check_value_for_legal_type(value=value, key=key)
            if not is_valid_type:
                problematic_keys[sample_name] = {key: {'value': value}}

    return problematic_keys


def check_value_for_legal_type(value, key: str):
    # TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists;
    # found <class 'NoneType'>
    if value is None:
        return False
    else:
        # Check that this actually catches the valid types, or has someone written a more bulletproof function
        if isinstance(value, (torch.Tensor, str, int, float, np.ndarray, dict, list)):
            return True
        else:
            return False


def ml_test_dataset_summary(ml_test_dataset: dict,
                            metric_key: str = 'samples_not_valid',
                            boolean_key: str = 'is_dataset_valid'):

    report_string = 'HEADER PLACEHOLDER\n'
    all_tests_ok = True
    for fold_name in ml_test_dataset:
        report_string += f'{fold_name}\n'
        for dataset_name in ml_test_dataset[fold_name]:
            report_string += f'\t{dataset_name}\n'
            for split_name in ml_test_dataset[fold_name][dataset_name]:
                report_string += f'\t\t{split_name}\n'
                metrics = ml_test_dataset[fold_name][dataset_name][split_name][metric_key]
                boolean = ml_test_dataset[fold_name][dataset_name][split_name][boolean_key]
                all_tests_ok = all_tests_ok and boolean

                # TO-OPTIMIZE! integrate with "add_boolean_and_metric_strings_to_summary()"
                for sample_name in metrics:
                    report_string += f'\t\t\t{sample_name}\n'
                    for key in metrics[sample_name]:
                        report_string += f'\t\t\t\t{key} = {metrics[sample_name][key]}\n'

    # e.g. report_string
    # HEADER PLACEHOLDER
    # fold0
    # 	MINIVESS
    # 		TRAIN
    # 			mv16
    # 				filepath_json = {'value': None}
    # 		VAL
    # 			mv38
    # 				filepath_json = {'value': None}
    # 		TEST
    # 			mv49
    # 				filepath_json = {'value': None}

    return all_tests_ok, report_string
