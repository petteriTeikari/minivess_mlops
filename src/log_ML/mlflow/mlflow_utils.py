import numpy as np
from loguru import logger
from mlflow.models import ModelSignature
from monai.utils import convert_to_tensor
from mlflow.types.schema import Schema, TensorSpec, ParamSchema, ParamSpec

from src.inference.ensemble_model import ModelEnsemble
from src.utils.train_utils import get_first_batch_from_dataloaders_dict


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

    # Inferring the input_output relationship
    model_output = define_model_input_and_output_shapes(image_tensor=batch_data['image'],
                                                        model_in=model_in,
                                                        return_mask=True)

    # Define the signature
    signature = define_model_schema(model_input=batch_data['image'],
                                    model_output=model_output)

    logger.info('MLflow | ModelSignature input Schema: {}'.format(signature.inputs))
    logger.info('MLflow | ModelSignature output Schema: {}'.format(signature.inputs))

    return signature


def define_model_schema(model_input,
                        model_output):

    # https://mlflow.org/docs/latest/models.html#id33

    # https://mlflow.org/docs/latest/models.html#id6
    # Enforce only the number of channels to be fixed 1, as the rest can be variable
    input_schema = Schema(
        [
            TensorSpec(np.dtype(np.float32), (-1, 1, -1, -1, -1)),
        ]
    )

    # for a binary mask output
    # TODO! how to return a dictionary over serving API?
    output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 1, -1, -1, -1))])

    params_schema = ParamSchema(
        [
            ParamSpec("return_mask", "boolean", True)
        ]
    )

    signature = ModelSignature(inputs=input_schema,
                               outputs=output_schema,
                               params=params_schema)

    return signature


def define_model_input_and_output_shapes(image_tensor,
                                         model_in: ModelEnsemble,
                                         return_mask: bool = False):

    tensor_input = convert_to_tensor(image_tensor)
    numpy_input = tensor_input.numpy()
    model_output: dict = model_in.predict(None,
                                          model_input=numpy_input,
                                          param={'return_mask': return_mask})

    return model_output


def mlflow_dicts_to_omegaconf_dict(experiment, active_run):

    def convert_indiv_dict(object_in, prefix: str = None):
        dict_out = {}
        for k, value in vars(object_in).items():

            if k[0] == '_':  # remove the _
                k = k[1:]

            if prefix is not None:
                key_out = prefix + k
            else:
                key_out = k

            # at the moment, just output the string values, as in names and paths
            if isinstance(value, str):
                dict_out[key_out] = value

        return dict_out

    experiment_out = convert_indiv_dict(object_in=experiment)
    mlflow_dict_out = {**experiment_out, **active_run.data.tags}

    return mlflow_dict_out