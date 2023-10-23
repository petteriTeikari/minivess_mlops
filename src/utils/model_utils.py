from collections import OrderedDict
from copy import deepcopy

import numpy as np
import torch
from loguru import logger

from monai.networks.nets import SegResNet, UNet
from torch import nn


def import_segmentation_model(model_config: dict, archi_name: str, device):

    # FIXME! Definitely want to support any Monai model out of the box, so parse the class name from string
    #  and do not map all possible models here, use like "automagic Monai" model use, and then allow the use
    #  of custom 3rd party models that you maybe found from Github or something

    if 'Unet' in archi_name:
        model = UNet(**model_config).to(device)
    elif archi_name == 'SegResNet':
        model = SegResNet(**model_config).to(device)
    else:
        raise NotImplementedError('Not implemented {} yet'.format(archi_name))

    logger.info('Using "{}" as the model, with the following parameters:'.format(archi_name))
    for param_key in model_config:
        logger.info(' {} = {}'.format(param_key, model_config[param_key]))

    return model


def get_nested_layers(model):
    # https://stackoverflow.com/a/65112132
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
        # look for children from children... to the last child!
        for child in children:
            try:
                flatt_children.extend(get_nested_layers(child))
            except TypeError:
                flatt_children.append(get_nested_layers(child))
    return flatt_children


def filter_layer_names(layers, layer_name_wildcard: str):

    layers_out = []
    for layer in layers:
        layer_name = layer.__class__.__name__
        if layer_name_wildcard in layer_name.lower():
            layers_out.append(deepcopy(layer))

    return layers_out


def get_layer_names_as_list(layers: list) -> list:
    layer_names = []
    for i, layer in enumerate(layers):
        layer_names.append(layer.__class__.__name__)
    return layer_names


def print_layer_names(layers: list):
    for i, layer in enumerate(layers):
        logger.debug(' {}/{}: {}'.format(i+1, len(layers), layer.__class__.__name__))


def get_last_layer_weights_of_model(model,
                                    p_weights: float = 1.00,
                                    layer_name_wildcard: str = 'conv'):

    # Get the last layer of the model with the wildcard
    weights = None
    layers = get_nested_layers(model)
    layers2 = filter_layer_names(layers, layer_name_wildcard=layer_name_wildcard)
    if len(layers2) > 0:
        last_layer = layers2[-1]
    else:
        logger.warning('Problem getting the last layer with wildcard "{}", layer names:'.
                       format(layer_name_wildcard))
        print_layer_names(layers)
        last_layer = None

    # Get the weights of the obtained last layer
    if last_layer is not None:
        try:
            weights_tensor = torch.squeeze(last_layer.weight.data[0])
            weights = np.copy(weights_tensor.detach().cpu().numpy())
        except Exception as e:
            # if you start having some more exotic models?
            raise IOError('Problem extracting the weights from the layer, e = {}'.format(e))

    # If you a lot features here, you might want to return only a subset of this
    # as these are atm used for ML testing purposes to check that you get the same
    # weights when loading models from disk / Model Registry as you got during the training
    if p_weights < 1:
        logger.warning('feature subset not implemented yet, '
                       'returning all the {} weights'.format(len(weights)))

    return weights


def create_pseudomodel_from_filtered_layers(layers: list):

    def list_to_list_of_tuples(layers: list) -> list:
        list_of_tuples = []
        for i, layer in enumerate(layers):
            list_of_tuples.append((layer.__class__.__name__, layer))
        return list_of_tuples

    def list_of_layers_to_ordered_dict(layers: list) -> OrderedDict:
        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#sequential
        list_of_tuples = list_to_list_of_tuples(layers)
        dict_out = OrderedDict(list_of_tuples)
        return dict_out

    model = nn.Sequential(list_of_layers_to_ordered_dict(layers))

    return model
