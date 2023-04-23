import glob
import os

from monai.data import CacheDataset, Dataset
from loguru import logger
from tqdm import tqdm
import requests
import zipfile
import random

from ml_tests.check_files import check_file_listing


def download_and_extract_minivess_dataset(input_url: str, data_dir: str,
                                          dataset_name: str = 'minivess'):

    input_filename = input_url.split('/')[-1]
    input_extension = '.zip'
    local_path = os.path.join(data_dir, input_filename + input_extension)

    zip_out = os.path.join(data_dir, input_filename)
    os.makedirs(zip_out, exist_ok=True)

    if os.path.exists(local_path):
        # if the zipping failed, and only the folder was created, this obviously thinks that the
        # download and extraction went well. #TO-OPTIMIZE
        logger.info('Dataset "{}" has already been downloaded and extracted to "{}"', dataset_name, zip_out)
    else:
        raise NotImplementedError('Implement the download!')
        response = requests.get(input_url)  # making requests to server
        logger.info('Downloading dataset "{}" to "{}"', dataset_name, local_path)
        with open(local_path, "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)

        with zipfile.ZipFile(local_path, 'r') as zip_ref:
            zip_ref.extractall(zip_out)

        dir_size_mb = round(100 * get_dir_size(start_path=zip_out) / (1024 ** 2)) / 100
        logger.info('Extracted the zip file to "{}" (size on disk = {} MB)', zip_out, dir_size_mb)

    return zip_out


def quick_and_dirty_minivess_clean_of_specific_file(images, labels, metadata, fname_in: str = 'mv39'):
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


def get_minivess_filelisting(dataset_dir: str,
                             use_quick_and_dirty_rejection: bool = True,
                             import_library = 'nibabel'):

    images = glob.glob(os.path.join(dataset_dir, 'raw', '*.nii.gz'))
    labels = glob.glob(os.path.join(dataset_dir, 'seg', '*.nii.gz'))
    metadata = glob.glob(os.path.join(dataset_dir, 'json', '*.json'))

    assert len(images) == len(labels), 'number of images ({}) and ' \
                                       'labels ({}) should match!'.format(len(images), len(labels))
    assert len(metadata) == len(labels), 'number of images ({}) and ' \
                                         'labels ({}) should match!'.format(len(metadata), len(labels))

    if use_quick_and_dirty_rejection:
        images, labels, metadata = \
            quick_and_dirty_minivess_clean_of_specific_file(images, labels, metadata, fname_in='mv39')

    images = sorted(images)
    labels = sorted(labels)
    metadata = sorted(metadata)

    # Run test for data integrity, i.e. there are no corrupted files, and return the headers back too
    size_stats, info_of_files, problematic_files = \
        check_file_listing(list_of_files=images, import_library='nibabel')

    filelisting = {'images': images,
                   'labels': labels,
                   'metadata': metadata}

    # Note! info_of_files contains NifTI headers, and we would like to be able to dump this dict as .yaml
    #       for reproducability issues. If you want something specific from the header, add it here
    #       in .yaml friendly format
    dataset_stats = {'size_stats': size_stats,
                     'problematic_files': problematic_files}

    return filelisting, dataset_stats


def define_minivess_splits(filelisting, data_splits_config: dict, include_metadata: bool = True):

    # Create data_dicts for MONAI, implement something else here if you don't want to use MONAI
    if include_metadata:
        logger.info('Include the metadata .json file to the input data dictionary for dataset creation')
        data_dicts = [
            {'image': image_name, 'label': label_name, 'metadata': metadata_name}
            for image_name, label_name, metadata_name in zip(filelisting['images'], filelisting['labels'],
                                                             filelisting['metadata'])
        ]
    else:
        logger.info('Not including the metadata .json file to the input data dictionary for dataset creation')
        data_dicts = [
            {'image': image_name, 'label': label_name}
            for image_name, label_name in zip(filelisting['images'], filelisting['labels'])
        ]

    split_method = data_splits_config['NAME']
    if split_method == 'RANDOM':
        files_dict = get_random_splits_for_minivess(data_dicts, data_split_cfg=data_splits_config[split_method])
        # FIXME! quick and dirty placeholder for allowing cross-validation if needed, or you could bootstrap
        #  an ensemble model with different data on each fold
        folds_splits = {'fold0': files_dict}
    else:
        raise NotImplementedError('Only implemented random splits at this point, '
                                  'not "{}"'.format(data_splits_config['NAME']))

    return folds_splits


def get_random_splits_for_minivess(data_dicts: list, data_split_cfg: dict):

    # Split data for training and testing.
    random.Random(data_split_cfg['SEED']).shuffle(data_dicts)
    # FIXME! Put something more intelligent here later instead of hard-coded n for val and test
    split_val_test = 7 # int(len(data_dicts) * .1)
    split_train = len(data_dicts) - 2*split_val_test # int(len(data_dicts) * .8)

    assert (split_train + split_val_test * 2 == len(data_dicts)), \
        'you lost some images during splitting, due to the int() operation most likely?\n' \
        'n_train = {} + n_val = {} + n_test = {} should be {}, ' \
        'but was {}'.format(split_train, split_val_test, split_val_test,
                            len(data_dicts), split_train + split_val_test * 2)

    files_dict = {
        'TRAIN': data_dicts[:split_train],
        'VAL': data_dicts[split_train:split_train + split_val_test],
        'TEST': data_dicts[split_train + split_val_test:]
    }

    sum_out = 0
    for split in files_dict.keys():
        sum_out += len(files_dict[split])

    assert sum_out == len(data_dicts), 'for some reason you lost files when doing training splits,' \
                                       'n_input = {}, n_output = {}'.format(len(data_dicts), sum_out)

    return files_dict


def define_minivess_dataset(dataset_config: dict, split_file_dicts: dict, transforms: dict):

    datasets = {}
    for i, split in enumerate(transforms.keys()):
        datasets[split] = create_dataset_per_split(dataset_config=dataset_config,
                                                   split=split,
                                                   split_file_dict=split_file_dicts[split],
                                                   transforms_per_split=transforms[split])

    return datasets


def create_dataset_per_split(dataset_config: dict, split: str, split_file_dict: dict, transforms_per_split: dict):

    n_files = len(split_file_dict)
    ds_config = dataset_config['DATASET']
    pytorch_dataset_type = ds_config['NAME']

    if pytorch_dataset_type == 'MONAI_CACHEDATASET':

        ds = CacheDataset(data=split_file_dict,
                         transform=transforms_per_split,
                         cache_rate=ds_config[pytorch_dataset_type]['CACHE_RATE'],
                         num_workers=ds_config[pytorch_dataset_type]['NUM_WORKERS'])
        logger.info('Created MONAI CacheDataset, split = "{}" (n = {}, '
                    'keys in dict = {})', split, n_files, list(split_file_dict[0].keys()))

    elif pytorch_dataset_type == 'MONAI_DATASET':
        ds = Dataset(data=split_file_dict)
        logger.info('Created MONAI (uncached) Dataset, split = "{}" (n={}, '
                    'keys in dict = {})', split, n_files, list(split_file_dict[0].keys))

    else:
        raise NotImplementedError('Not implemented yet other dataset than Monai CacheDataset and Dataset, '
                                  'not = "{}"'.format(pytorch_dataset_type))

    return ds
