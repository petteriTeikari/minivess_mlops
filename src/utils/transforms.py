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
    EnsureTyped,
)

import numpy

def define_transforms(dataset_config: dict):

    transforms = {}
    for i, split in enumerate(dataset_config['TRANSFORMS'].keys()):
        transforms[split] = get_transforms_per_split(transformation_cfg = dataset_config['TRANSFORMS'][split],
                                                     split = split)

    return transforms


def get_transforms_per_split(transformation_cfg: list, split: str = 'TRAIN', np=None):

    # TOADD! Later actually use the list from the config, now just pass whatever basic transformations
    #        to debug the script
    # TOADD! encode to transforms also the library? e.g. using transforms from MONAI or from Pytorch?
    if split == 'TRAIN':
        transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys=['image', 'label'], minv=0.0, maxv=1.0),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 1),  # (160, 160),
                    pos=3,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 1),
                    rotate_range=(0, 0, numpy.pi / 15),
                    scale_range=(0.1, 0.1)),
                EnsureTyped(keys=["image", "label"], data_type='tensor'),
            ]
        )
    elif split == 'VAL' or split == 'TEST':
        transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                ScaleIntensityd(keys=['image', 'label'], minv=0.0, maxv=1.0),
                CenterSpatialCropd(keys=["image", "label"], roi_size=[-1, -1, 1]),
                EnsureTyped(keys=["image", "label"], data_type='tensor'),
            ]
        )
    else:
        raise NotImplementedError('Unknown split = {} (should be TRAIN, VAL or TEST)'.format(split))

    return transforms


