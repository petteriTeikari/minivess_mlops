import mlflow
import torch
from loguru import logger
from mlflow.models import infer_signature
from monai.utils import convert_to_tensor

from src.inference.ensemble_model import ModelEnsemble
from src.utils.train_utils import get_first_batch_from_dataloaders_dict


def get_model_from_mlflow_model_registry(model_uri):
    """
    not https://mlflow.org/docs/latest/model-registry.html#fetching-an-mlflow-model-from-the-model-registry
    pyfunc_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    See this
    https://mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.load_model
    """
    logger.info('MLflow | Fetching Pytorch model from Model Registry: {}'.format(model_uri))
    loaded_model = mlflow.pytorch.load_model(model_uri)

    return loaded_model


def get_subdicts_from_mlflow_model_log(mlflow_model_log: dict, key_str: str):

    subdicts = {}
    for submodel_name in mlflow_model_log:
        subdicts[submodel_name] = mlflow_model_log[submodel_name][key_str]

    return subdicts


def get_mlflow_model_signature_from_dataloader_dict(model_in: ModelEnsemble,
                                                    experim_dataloaders: dict):
    """
    https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.ModelSignature
    """

    # Get a sample batch from the dataloder for the signature
    batch_data = get_first_batch_from_dataloaders_dict(experim_dataloaders=experim_dataloaders)

    # Inferring the input signature
    tensor_input = convert_to_tensor(batch_data['image'])
    numpy_input = tensor_input.numpy()
    batch_sz, no_channels, dim1, dim2, dimz = numpy_input.shape
    numpy_input = numpy_input[0, :, :, :]
    no_chans, dim1, dim2, dimz = numpy_input.shape
    model_output = model_in.predict_single_volume(image_tensor=tensor_input,
                                                  input_as_numpy=True,
                                                  return_mask=True)
    if isinstance(model_output, torch.Tensor):
        model_output = model_output.detach().cpu().numpy()

    logger.info('MLflow | Obtaining MLflow model signature (input.shape = {}, output_shape = {})'.
                format(numpy_input.shape, model_output.shape))
    signature = infer_signature(model_input=numpy_input,
                                model_output=model_output)
    # TO-OPTIMIZE! Check how these should be when you start using this from MLflow Models
    # none of the spatial dims are fixed obviously (96x96x8 just the size now used in the MONAI
    # inference routine)
    logger.info('MLflow | ModelSignature input Schema: {}'.format(signature.inputs))
    logger.info('MLflow | ModelSignature output Schema: {}'.format(signature.inputs))

    return signature
