from loguru import logger
from monai.data import CacheDataset
from src.utils.general_utils import print_memory_stats_to_logger
from tests.data.dataset_tests import ml_test_dataset_for_allowed_types


def define_torch_dataset(
    dataset_name: str,
    dataset_config: dict,
    split_file_dicts: dict,
    transforms: dict,
    debug_testing: bool,
):
    datasets, ml_test_dataset = {}, {}
    print_memory_stats_to_logger()
    for i, split in enumerate(transforms.keys()):
        datasets[split], ml_test_dataset[split] = create_dataset_per_split(
            dataset_config=dataset_config,
            split=split,
            split_file_dict=split_file_dicts[split],
            transforms_per_split=transforms[split],
            debug_testing=debug_testing,
        )

        # Print the available memory after each dataset creation (you might run out of memory on your machine
        # if you "cache too much" on a machine with not enough RAM)
        print_memory_stats_to_logger()

    return datasets, ml_test_dataset


def create_dataset_per_split(
    dataset_config: dict,
    split: str,
    split_file_dict: dict,
    transforms_per_split: dict,
    debug_testing: bool = False,
):
    n_files = len(split_file_dict)
    ds_config = dataset_config["DATASET"]
    pytorch_dataset_type = ds_config["NAME"]

    if debug_testing:
        split_file_dict = debug_add_errors_to_dataset_dict(split_file_dict)

    is_dataset_valid, samples_not_valid = ml_test_dataset_for_allowed_types(
        split_file_dict=split_file_dict
    )

    if pytorch_dataset_type == "MONAI_CACHEDATASET":
        # TODO! for fancier RAM management, you could adaptively set the cache_rate here
        #  based on the machine that you are running this on, add like a case with
        #  "if ds_config[pytorch_dataset_type]['CACHE_RATE'] == 'max_avail'"
        ds = CacheDataset(
            data=split_file_dict,
            transform=transforms_per_split,
            cache_rate=ds_config[pytorch_dataset_type]["CACHE_RATE"],
            num_workers=ds_config[pytorch_dataset_type]["NUM_WORKERS"],
        )

        logger.info(
            'Created MONAI CacheDataset, split = "{}" (n = {}, '
            "keys in dict = {}, cache_rate = {}, num_workers = {})",
            split,
            n_files,
            list(split_file_dict[0].keys()),
            ds_config[pytorch_dataset_type]["CACHE_RATE"],
            ds_config[pytorch_dataset_type]["NUM_WORKERS"],
        )

    elif pytorch_dataset_type == "MONAI_DATASET":
        logger.error(
            "WARNING! You are using the vanilla MONAI Dataset, which does not work downstream from here"
        )
        raise NotImplementedError(
            "Vanilla MONAI dataset not implemented, use CacheDataset instead with "
            "cache_rate=0 if you have issues with RAM availability on your machine"
        )
    else:
        raise NotImplementedError(
            "Not implemented yet other dataset than Monai CacheDataset and Dataset, "
            'not = "{}"'.format(pytorch_dataset_type)
        )

    ml_test_dataset = {
        "is_dataset_valid": is_dataset_valid,
        "samples_not_valid": samples_not_valid,
    }

    return ds, ml_test_dataset


def debug_add_errors_to_dataset_dict(split_file_dict: dict):
    logger.warning(
        "WARNING You are intentionally adding errors to our dataset for testing the ML Tests pipeline"
    )
    # TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists;
    # found <class 'NoneType'>
    split_file_dict[0]["metadata"]["filepath_json"] = None

    return split_file_dict


def get_sample_keys(split_file_dicts: dict) -> list:
    split_keys = list(split_file_dicts.keys())
    samples_per_split = split_file_dicts[split_keys[0]]
    first_sample = samples_per_split[0]
    sample_keys = list(first_sample.keys())
    return sample_keys
