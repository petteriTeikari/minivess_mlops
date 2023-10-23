from copy import deepcopy
from typing import Union

import mlflow
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from loguru import logger

from src.inference.ensemble_utils import add_sample_results_to_ensemble_results, add_sample_metrics_to_split_results, \
    compute_split_metric_stats
from src.inference.inference_utils import inference_sample, get_inference_metrics
from src.log_ML.model_saving import import_model_from_path
from src.utils.model_utils import get_last_layer_weights_of_model


def inference_ensemble_dataloader(models_of_ensemble: dict,
                                  config: dict,
                                  split: str,
                                  dataloader,
                                  device):

    # Create the ensembleModel class with all the submodels of the ensemble
    ensemble_model = ModelEnsemble(models_of_ensemble=models_of_ensemble,
                                   validation_config=config['config']['VALIDATION'],
                                   ensemble_params=config['config']['ENSEMBLE']['PARAMS'],
                                   validation_params=config['config']['VALIDATION']['VALIDATION_PARAMS'],
                                   device=device,
                                   eval_config=config['config']['VALIDATION_BEST'],
                                   # TODO! this could be architecture-specific
                                   precision='AMP')  # config['config']['TRAINING']['PRECISION'])

    ensemble_results = inference_ensemble_with_dataloader(ensemble_model,
                                                          dataloader=dataloader,
                                                          split=split)

    return ensemble_results


def inference_ensemble_with_dataloader(ensemble_model,
                                       dataloader,
                                       split: str):

    with (torch.no_grad()):
        dataloader_metrics = {}
        for batch_idx, batch_data in enumerate(
                tqdm(dataloader,
                     desc='ENSEMBLE: Inference on dataloader samples, split "{}"'.format(split),
                     position=0)):
            dict_out = ensemble_model.predict_with_gt(batch_data=batch_data)
            sample_ensemble_metrics = dict_out['sample_ensemble_metrics']

            # Collect the metrics for each sample so you can for example compute mean dice for the split
            dataloader_metrics = \
                add_sample_metrics_to_split_results(sample_metrics=deepcopy(sample_ensemble_metrics),
                                                    dataloader_metrics_tmp=dataloader_metrics)

    # Done with the dataloader here and you have metrics computed per each of the sample
    # in the dataloader and you would like to probably have like mean Dice and stdev in Dice
    # along with the individual values so you could plot some distribution and highlight the
    # poorly performing samples
    dataloader_metrics_stat = compute_split_metric_stats(dataloader_metrics=dataloader_metrics)

    return {'samples': dataloader_metrics, 'stats': dataloader_metrics_stat}


def get_monai_param_dict_from_omegaconf(validation_params, model) -> dict:
    monai_metric_dict = OmegaConf.to_container(validation_params)  # OmegaConf -> Dict
    monai_metric_dict['predictor'] = model
    return monai_metric_dict


def ensemble_repeats(inf_res: dict,
                     ensemble_params: dict,
                     var_type_key: str = 'arrays',
                     var_key: str = 'probs') -> dict:

    if var_type_key in inf_res:
        if var_key in inf_res[var_type_key]:
            input_data = inf_res[var_type_key][var_key]
        else:
            logger.error('var_key = {}, not found in the var_type_key = {}, keys = {}'.
                         format(var_key, var_type_key, inf_res[var_type_key].keys()))
            raise IOError('var_key = {}, not found in the var_type_key = {}, keys = {}'.
                          format(var_key, var_type_key, inf_res[var_type_key].keys()))
    else:
        logger.error('var_type_key = {}, not in the inference results, keys = {}'.
                     format(var_type_key, inf_res.keys()))
        raise IOError('var_type_key = {}, not in the inference results, keys = {}'.
                      format(var_type_key, inf_res.keys()))

    ensemble_stats = compute_ensembled_response(input_data, ensemble_params)

    return ensemble_stats


def compute_ensembled_response(input_data: np.ndarray,
                               ensemble_params) -> dict:

    variable_stats = {'scalars': {}}
    variable_stats['scalars']['n_samples'] = input_data.shape[0]  # i.e. number of repeats / submodels

    variable_stats = {'arrays': {}}
    variable_stats['arrays']['mean'] = np.mean(input_data, axis=0)  # i.e. number of repeats / submodels
    variable_stats['arrays']['var'] = np.var(input_data, axis=0)  # i.e. number of repeats / submodels
    variable_stats['arrays']['UQ_epistemic'] = np.nan
    variable_stats['arrays']['UQ_aleatoric'] = np.nan
    variable_stats['arrays']['entropy'] = np.nan

    if ensemble_params['method'] == 'average':
        variable_stats['arrays']['mask'] = (
            (variable_stats['arrays']['mean'] > ensemble_params['mask_threshold']).astype('float32'))
    else:
        # majority_voting_here
        raise NotImplementedError('Unknown ensemble method = {}'.format(ensemble_params['method']))

    return variable_stats


class ModelEnsemble(mlflow.pyfunc.PythonModel):
    """
    To facilitate the use of Model Registries for ensembles and in general reproducibility of your stuff from MLflow
    https://medium.com/@pennyqxr/how-to-train-and-track-ensemble-models-with-mlflow-a1d2695e784b
    https://medium.com/p/166aeb8666a8
    """
    def __init__(self, models_of_ensemble: dict, validation_config: dict,
                 ensemble_params: dict, validation_params: dict,
                 device, precision: str, eval_config: dict, model_best_dicts: dict = None,
                 models_from_paths: bool = True, mode: str = 'evaluate'):
        self.models_of_ensemble = models_of_ensemble
        self.no_submodels = len(models_of_ensemble)
        if model_best_dicts is None:
            self.model_best_dicts = {}
        else:
            self.model_best_dicts = model_best_dicts
        self.models = {}
        self.ensemble_params = ensemble_params
        self.device = device
        self.validation_params = validation_params
        self.precision = precision
        self.eval_config = eval_config

        for submodel_name in models_of_ensemble:

            if models_from_paths:
                # During training, you have the local model paths that you are loading the model from
                self.models[submodel_name], self.model_best_dicts[submodel_name], _, _ = (
                    import_model_from_path(model_path=models_of_ensemble[submodel_name],
                                           validation_config=validation_config))
            else:
                # you already have the models loaded as in the case with MLflow model registry fetch
                self.models[submodel_name] = models_of_ensemble[submodel_name]
                if len(self.model_best_dicts) == 0:
                    raise IOError('Your best_dicts is empty, refactor this to be smarter,'
                                  'as with MLflow registry load, you provide this and do not load'
                                  'from the saved model')

            if mode == 'evaluate':
                self.models[submodel_name].eval()
            else:
                raise NotImplementedError('Only evaluation defined, fix here if you want to resume training')

    def __inference_per_sample(self,
                               input_data,
                               input_from: str = 'dict'):

        # Inference the batch_data through all the submodels
        inference_results = {}
        for submodel_name in self.models:
            model = self.models[submodel_name]
            monai_metric_dict = get_monai_param_dict_from_omegaconf(self.validation_params, model)
            sample_res = inference_sample(input_data,
                                          metric_dict=monai_metric_dict,
                                          device=self.device,
                                          input_from=input_from,
                                          precision=self.precision)

            # Add different repeats together so you can get the ensemble response
            inference_results = add_sample_results_to_ensemble_results(inf_res=deepcopy(inference_results),
                                                                       sample_res=sample_res)

        return inference_results

    def predict_single_volume(self,
                              image_tensor: Union[np.ndarray, torch.Tensor],
                              input_as_numpy: bool = False,
                              return_mask: bool = True,
                              add_channel_to_output: bool = True):
        """
        When you only have the tensor of the images and no idea of the ground truth,
        i.e. when you want to use the trained model for actual segmentation
        """
        if input_as_numpy:
            image_tensor = torch.Tensor(image_tensor)

        inference_results = self.__inference_per_sample(input_data=image_tensor,
                                                        input_from='tensor')

        ensemble_stat_results = ensemble_repeats(inf_res=inference_results,
                                                 ensemble_params=self.ensemble_params)

        if return_mask:
            mask_out = ensemble_stat_results['arrays']['mask']
            if add_channel_to_output:
                if isinstance(mask_out, np.ndarray):
                    mask_out = mask_out[np.newaxis, :, :, :]
                else:
                    raise NotImplementedError('Check what are the dims for torch.Tensor output')
            return mask_out
        else:
            # This contains pixel-wise probabilities, uncertainty estimates, etc in a dictionary
            return ensemble_stat_results

    def predict_with_gt(self,
                        batch_data,
                        input_from: str = 'dict',
                        return_mask: bool = False):
        """
        You can use this during the training when you know the ground truth,
        and the batch_data is the dictionary from the dataloader
        """
        inference_results = self.__inference_per_sample(input_data=batch_data,
                                                        input_from='dict')

        # We have now the inference output of each submodel in the "ensemble_results" and we can for example
        # get the average probabilities per pixel/voxel, or do majority voting
        # This contains the binary mask that you could actually use
        ensemble_stat_results = ensemble_repeats(inf_res=inference_results,
                                                 ensemble_params=self.ensemble_params)

        # And let's compute the metrics from the ensembled prediction
        sample_ensemble_metrics = (
            get_inference_metrics(ensemble_stat_results=ensemble_stat_results,
                                  y_pred=ensemble_stat_results['arrays']['mask'],
                                  eval_config=self.eval_config,
                                  batch_data=batch_data))

        dict_out = {'inference_results': inference_results,
                    'ensemble_stat_results': ensemble_stat_results,
                    'sample_ensemble_metrics': sample_ensemble_metrics}

        if return_mask:
            return dict_out['ensemble_stat_results']['arrays']['mask']
        else:
            return dict_out

    def get_model_weights(self):

        ensemble_weights = []
        for ensemble_name in self.models:
            model = self.models[ensemble_name]
            weights = get_last_layer_weights_of_model(model,
                                                      p_weights=1.00,
                                                      layer_name_wildcard='conv')
            ensemble_weights.append(weights)

        return ensemble_weights
