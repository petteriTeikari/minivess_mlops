import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete

def inference_sample(batch_data,
                     model,
                     metric_dict: dict,
                     device: str,
                     auto_mixedprec: bool = True):
    """
    See e.g. https://docs.monai.io/en/stable/inferers.html
             https://github.com/davidiommi/Ensemble-Segmentation/blob/main/predict_single_image.py
             https://github.com/Project-MONAI/tutorials/blob/main/modules/cross_validation_models_ensemble.ipynb
    :param model:
    :param dataloader:
    :param config:
    :param device:
    :param auto_mixedprec:
    :return:
    """
    if auto_mixedprec:  ## AMP
        with torch.cuda.amp.autocast():
            output = sliding_window_inference(inputs=batch_data["image"].to(device), **metric_dict)
    else:
        output = sliding_window_inference(inputs=batch_data["image"].to(device), **metric_dict)

    # logits -> probabilities
    activ = Activations(sigmoid=True)
    probability_mask = activ(output)

    # probabilities -> binary mask
    discret = AsDiscrete(threshold=0.5)
    binary_mask = discret(probability_mask)

    inference_output = {'scalars':
                            {
                                'dummy_scalar': np.array([0.5])
                            },
                        'arrays':
                            {
                                #'logits': conv_metatensor_to_numpy(output),
                                'probs': conv_metatensor_to_numpy(probability_mask)
                                #'binary_mask': conv_metatensor_to_numpy(binary_mask)
                            },
                        }

    return inference_output


def conv_metatensor_to_numpy(metatensor_in: MetaTensor) -> np.ndarray:
    # https://github.com/Project-MONAI/MONAI/issues/4649#issuecomment-1177319399
    numpy_array = metatensor_in.detach().numpy()
    assert len(numpy_array.shape) == 5, ('Code tested only for 5D tensors at the moment, '
                                         'now your metatensor is {}D').format(len(numpy_array.shape))
    return numpy_array[0,0,:,:,:]  # quick and dirty for single-channel and batch_sz = 1 tensors


