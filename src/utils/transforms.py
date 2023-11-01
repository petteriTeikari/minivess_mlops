from loguru import logger
from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    EnsureChannelFirstd,
    EnsureType,
    CenterSpatialCropd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RemoveSmallObjects,
    ScaleIntensityd,
    EnsureTyped, ToDeviced,
)

import numpy


def define_transforms(dataset_config: dict,
                      transform_config_per_dataset: dict,
                      transform_config: dict,
                      device,
                      keys_in_samples: list):

    transforms = {}
    for i, split in enumerate(transform_config_per_dataset.keys()):
        transforms[split] = get_transforms_per_split(device=device,
                                                     transform_cfg_name_per_split=transform_config_per_dataset[split],
                                                     split=split,
                                                     transform_config=transform_config,
                                                     keys_in_samples=keys_in_samples)

    return transforms


def get_transforms_per_split(device,
                             transform_cfg_name_per_split: str,
                             split: str,
                             transform_config: dict,
                             keys_in_samples: list):

    if transform_cfg_name_per_split == 'BASIC_AUG':
        transforms = basic_aug(device)  # eliminate the if/elif/else and parse directly from str?
    elif transform_cfg_name_per_split == 'NO_AUG':
        if 'label' in keys_in_samples:
            transforms = no_aug(device)
        else:
            logger.info('No label found in the Dataset, you are running inference?')
            transforms = no_aug(device, for_inference_without_labels=True)
    else:
        raise NotImplementedError('Not implemented yet the desired "{}" transform'.
                                  format(transform_cfg_name_per_split))

    return transforms


def basic_aug(device):

    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            # ToDeviced(keys=["image", "label"], device=device), # torch.cuda.OutOfMemoryError: CUDA out of memory.
            EnsureChannelFirstd(keys=["image", "label"]),
            ScaleIntensityd(keys=['image', 'label'], minv=0.0, maxv=1.0),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            # FIXME! Add some padding operator here as we quick'n'dirty rejected the one volume with only 5 slices
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 8),  # (160, 160),
                pos=3,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=1.0, spatial_size=(96, 96, 8),
                rotate_range=(0, 0, numpy.pi / 15),
                scale_range=(0.1, 0.1)),
            EnsureTyped(keys=["image", "label"], data_type='tensor'),
        ]
    )

    return transforms


def no_aug(device: str,
           for_inference: bool = False,
           for_inference_without_labels: bool = False):
    # Inference without labels
    if for_inference_without_labels:
        transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                # ToDeviced(keys=["image"], device=device),
                EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityd(keys=['image'], minv=0.0, maxv=1.0),
                EnsureTyped(keys=["image"], data_type='tensor')
            ]
        )
    # Inference with labels
    elif for_inference:
        # no random cropping, doing inference for the original shape with a batch size of 1
        transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                # ToDeviced(keys=["image", "label"], device=device),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys=['image', 'label'], minv=0.0, maxv=1.0),
                EnsureTyped(keys=["image", "label"], data_type='tensor')
            ]
        )
    # Evaluation (the smaller crops)
    else:
        transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                # ToDeviced(keys=["image", "label"], device=device),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys=['image', 'label'], minv=0.0, maxv=1.0),
                # problem that not all volumes have the same shape
                # CenterSpatialCropd(keys=["image", "label"], roi_size=[-1, -1, -1]),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(128, 128, 8),
                    pos=3,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                EnsureTyped(keys=["image", "label"], data_type='tensor'),
            ]
        )

    return transforms
