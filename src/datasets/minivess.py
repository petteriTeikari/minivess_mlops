import glob
import os
from loguru import logger
import requests
import zipfile
import random

from omegaconf import DictConfig

from src.datasets.torch_dataset import create_dataset_per_split
from src.utils.dict_utils import cfg_key
from tests.data.file_tests import ml_test_filelisting_corruption
from src.datasets.dvc_utils import get_dvc_files_of_repo
from src.utils.general_utils import print_memory_stats_to_logger


def import_minivess_dataset(
    dataset_cfg: DictConfig,
    data_dir: str,
    run_mode: str,
    cfg: dict,
    dataset_name: str,
    fetch_method: str,
    fetch_params: dict,
):
    if fetch_method == "DVC":
        repo_url = cfg_key(cfg, "hydra_cfg", "config", "SERVICES", "DVC", "repo_url")
        repo_dir = cfg_key(cfg, "run", "PARAMS", "repo_dir")
        dataset_dir = fetch_dataset_with_dvc(
            fetch_params=fetch_params,
            dataset_cfg=dataset_cfg,
            dataset_name_lowercase=dataset_name.lower(),
            repo_url=repo_url,
            repo_dir=repo_dir,
        )

    elif fetch_method == "EBrains":
        raise NotImplementedError("EBrains not working atm, use DVC instead")
        # dataset_dir = import_filelisting_ebrains(dataset_cfg=dataset_cfg,
        #                                          data_dir=data_dir,
        #                                          fetch_params=fetch_params)

    else:
        raise NotImplementedError(
            "Unknown dataset fetch method for Minivess = {}".format(fetch_method)
        )

    filelisting, dataset_stats = get_minivess_filelisting(dataset_dir)
    fold_split_file_dicts = define_minivess_splits(
        filelisting, data_splits_config=dataset_cfg["SPLITS"]
    )

    try:
        if dataset_cfg["SUBSET"]["NAME"] == "ALL_SAMPLES":
            logger.info("Using all the samples for training, validation and testing")
        else:
            subset_cfg_name = dataset_cfg["SUBSET"]["NAME"]
            subset_cfg = dataset_cfg["SUBSET"][subset_cfg_name]
            logger.info(
                "Using only a subset of the samples, Config = {}".format(
                    dataset_cfg["SUBSET"]["NAME"]
                )
            )
            minivess_debug_splits(
                fold_split_file_dicts=fold_split_file_dicts,
                subset_cfg=subset_cfg,
                subset_cfg_name=subset_cfg_name,
            )
    except Exception as e:
        logger.error("Problem getting data subset, error = {}".format(e))
        raise IOError("Problem getting data subset, error = {}".format(e))

    return filelisting, fold_split_file_dicts, dataset_stats


def fetch_dataset_with_dvc(
    fetch_params: dict,
    dataset_cfg: DictConfig,
    dataset_name_lowercase: str,
    repo_url: str,
    repo_dir: str,
):
    # TODO! This is now based on "dvc pull" by Github Actions or manual, but you could try
    #  to get the Python programmatic API to work too (or have "dvc pull" from subprocess)
    dataset_dir = get_dvc_files_of_repo(
        repo_dir=repo_dir,
        repo_url=repo_url,
        dataset_name_lowercase=dataset_name_lowercase,
        fetch_params=fetch_params,
        dataset_cfg=dataset_cfg,
    )

    return dataset_dir


def import_filelisting_ebrains(dataset_cfg: dict, data_dir: str, fetch_params: dict):
    logger.warning(
        "Work on the EBrains method more if you want to use this,"
        "recommended to use DVC for now"
    )
    input_url = dataset_cfg["DATA_DOWNLOAD_URL"]
    dataset_dir = download_and_extract_minivess_dataset(
        input_url=input_url, data_dir=data_dir
    )

    return dataset_dir


def download_and_extract_minivess_dataset(
    input_url: str, data_dir: str, dataset_name: str = "minivess"
):
    def extract_minivess_zip(local_zip_path: str, local_data_dir: str):
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            zip_ref.extractall(local_data_dir)

    def check_if_extracted_already(local_data_dir: str) -> bool:
        already_extracted = False
        if os.path.exists(local_data_dir):
            files_and_subdirs_in_dir = glob.glob(os.path.join(local_data_dir, "*"))
            if len(files_and_subdirs_in_dir) > 0:
                already_extracted = True
        return already_extracted

    def download_zip(input_url: str):
        query_parameters = {"downloadformat": "zip"}
        response = requests.get(input_url, params=query_parameters)
        with open("gdp_by_country.zip", mode="wb") as file:
            ...
            file.write(response.content)

    input_filename = input_url.split("/")[-1]
    input_extension = ".zip"
    local_data_dir = os.path.join(data_dir, input_filename)
    local_zip_path = local_data_dir + input_extension

    zip_out = os.path.join(data_dir, input_filename)
    os.makedirs(zip_out, exist_ok=True)

    if check_if_extracted_already(local_data_dir):
        logger.info(
            'Dataset "{}" has already been downloaded and extracted to "{}"',
            dataset_name,
            zip_out,
        )

    elif os.path.exists(local_zip_path):
        # you have manually pasted the zip here and this would autoextract the files to the three folders,
        # or your download went fine, and something crashed before extraction
        logger.info(
            'Dataset .zip "{}" has already been downloaded, and extracting this file "{}"',
            dataset_name,
            zip_out,
        )
        extract_minivess_zip(local_zip_path, local_data_dir)

    else:
        logger.info('Downloading dataset "{}" to "{}"', dataset_name, local_zip_path)
        download_zip_from_ebrains(input_url, local_zip_path)
        extract_minivess_zip(local_zip_path, local_data_dir)
        # dir_size_mb = round(100 * get_dir_size(start_path=zip_out) / (1024 ** 2)) / 100
        # logger.info('Extracted the zip file to "{}" (size on disk = {} MB)', zip_out, dir_size_mb)

    return zip_out


def download_zip_from_ebrains(input_url, local_zip_path):
    """
    https://github.com/HumanBrainProject/ebrains-drive/blob/master/doc.md#seaffile_get
    https://gitlab.ebrains.eu/fousekjan/tvb-ebrains-data/-/blob/master/tvb_ebrains_data/data.py
    https://github.com/HumanBrainProject/ebrains-drive
    https://github.com/HumanBrainProject/fairgraph

    Not sure if the download is possible unauthenticated?
    """

    raise NotImplementedError(
        "Implement the download from EBRAINS. your standard requests approach will not work\n"
        "Manual workaround now by downloading the .zip manually: {}\n"
        "Save this then to {}\n"
        "And run this script again, and the .zip will be autoextracted".format(
            input_url, local_zip_path
        )
    )


def quick_and_dirty_minivess_clean_of_specific_file(
    images, labels, metadata, fname_in: str = "mv39"
):
    logger.info('Reject "{}.nii.gz" as it is only 512x512x5 voxels'.format(fname_in))
    n_in = len(images)
    image_path = [s for s in images if fname_in in s][0]
    label_path = [s for s in labels if fname_in in s][0]
    metadata_path = [s for s in metadata if fname_in in s][0]
    images.remove(image_path)
    labels.remove(label_path)
    metadata.remove(metadata_path)
    n_out = len(images)

    return images, labels, metadata


def get_minivess_filelisting(
    dataset_dir: str,
    use_quick_and_dirty_rejection: bool = True,
    import_library: str = "nibabel",
):
    images = glob.glob(os.path.join(dataset_dir, "raw", "*.nii.gz"))
    labels = glob.glob(os.path.join(dataset_dir, "seg", "*.nii.gz"))
    metadata = glob.glob(os.path.join(dataset_dir, "json", "*.json"))

    assert len(images) == len(
        labels
    ), "number of images ({}) and " "labels ({}) should match!".format(
        len(images), len(labels)
    )
    assert len(metadata) == len(
        labels
    ), "number of images ({}) and " "labels ({}) should match!".format(
        len(metadata), len(labels)
    )

    if use_quick_and_dirty_rejection:
        images, labels, metadata = quick_and_dirty_minivess_clean_of_specific_file(
            images, labels, metadata, fname_in="mv39"
        )

    images = sorted(images)
    labels = sorted(labels)
    metadata = sorted(metadata)

    # Run test for data integrity, i.e. there are no corrupted files, and return the headers back too
    size_stats, info_of_files, problematic_files = ml_test_filelisting_corruption(
        list_of_files=images, import_library="nibabel"
    )

    filelisting = {"images": images, "labels": labels, "metadata": metadata}

    # Note! info_of_files contains NifTI headers, and we would like to be able to dump this dict as .yaml
    #       for reproducability issues. If you want something specific from the header, add it here
    #       in .yaml friendly format
    dataset_stats = {"size_stats": size_stats, "problematic_files": problematic_files}

    return filelisting, dataset_stats


def define_minivess_splits(
    filelisting, data_splits_config: dict, include_metadata: bool = True
):
    # Create data_dicts for MONAI, implement something else here if you don't want to use MONAI
    if include_metadata:
        logger.info(
            "Include the metadata .json file to the input data dictionary for dataset creation"
        )
        data_dicts = [
            {
                "image": image_name,
                "label": label_name,
                "metadata": {"filepath_json": metadata_name},
            }
            for image_name, label_name, metadata_name in zip(
                filelisting["images"], filelisting["labels"], filelisting["metadata"]
            )
        ]
    else:
        logger.info(
            "Not including the metadata .json file to the input data dictionary for dataset creation"
        )
        data_dicts = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(
                filelisting["images"], filelisting["labels"]
            )
        ]

    split_method = data_splits_config["NAME"]
    if split_method == "RANDOM":
        logger.info("Splitting the data randomly")
        files_dict = get_random_splits_for_minivess(
            data_dicts, data_split_cfg=data_splits_config[split_method]
        )
        # FIXME! quick and dirty placeholder for allowing cross-validation if needed, or you could bootstrap
        #  an inference model with different data on each fold
        folds_splits = {"fold0": files_dict}
    else:
        raise NotImplementedError(
            "Only implemented random splits at this point, "
            'not "{}"'.format(data_splits_config["NAME"])
        )

    return folds_splits


def get_random_splits_for_minivess(data_dicts: list, data_split_cfg: dict):
    # Split data for training and testing.
    random.Random(data_split_cfg["SEED"]).shuffle(data_dicts)
    # FIXME! Put something more intelligent here later instead of hard-coded n for val and test
    split_val_test = 7  # int(len(data_dicts) * .1)
    split_train = len(data_dicts) - 2 * split_val_test  # int(len(data_dicts) * .8)

    assert split_train + split_val_test * 2 == len(data_dicts), (
        "you lost some images during splitting, due to the int() operation most likely?\n"
        "n_train = {} + n_val = {} + n_test = {} should be {}, "
        "but was {}".format(
            split_train,
            split_val_test,
            split_val_test,
            len(data_dicts),
            split_train + split_val_test * 2,
        )
    )

    files_dict = {
        "TRAIN": data_dicts[:split_train],
        "VAL": data_dicts[split_train : split_train + split_val_test],
        "TEST": data_dicts[split_train + split_val_test :],
    }

    sum_out = 0
    for split in files_dict.keys():
        sum_out += len(files_dict[split])

    assert sum_out == len(data_dicts), (
        "for some reason you lost files when doing training splits,"
        "n_input = {}, n_output = {}".format(len(data_dicts), sum_out)
    )

    return files_dict


def define_minivess_dataset(
    dataset_config: dict, split_file_dicts: dict, transforms: dict, debug_testing: bool
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


def minivess_debug_splits(
    fold_split_file_dicts: dict, subset_cfg: dict, subset_cfg_name: str
):
    logger.warning(
        "You are not using the full fileset now! (subset method = {})".format(
            subset_cfg_name
        )
    )
    for fold in fold_split_file_dicts:
        for split in fold_split_file_dicts[fold]:
            logger.warning(
                "First {} samples for split = {}".format(subset_cfg[split], split)
            )
            fold_split_file_dicts[fold][split] = fold_split_file_dicts[fold][split][
                0 : subset_cfg[split]
            ]

    return fold_split_file_dicts
