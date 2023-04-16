from loguru import logger

from monai.networks.nets import SegResNet, UNet

def import_segmentation_model(model_config: dict, device):

    # FIXME! Definitely want to support any Monai model out of the box, so parse the class name from string
    #  and do not map all possible models here, use like "automagic Monai" model use, and then allow the use
    #  of custom 3rd party models that you maybe found from Github or something
    if model_config['MODEL_NAME'] == 'Unet':
        model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=3,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        ).to(device)
    else:
        raise NotImplementedError('Not implemented {} yet'.format(model_config['MODEL_NAME']))

    logger.info('Using "{}" as the model, with the following parameters:'.format(model_config['MODEL_NAME']))
    params_cfg = model_config[model_config['MODEL_NAME']]
    for param_key in params_cfg:
        logger.info(' {} = {}'.format(param_key, params_cfg[param_key]))

    return model