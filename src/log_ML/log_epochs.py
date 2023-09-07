import os
import warnings
from loguru import logger

from torch.utils.tensorboard import SummaryWriter

from src.log_ML.log_metrics_lowlevel import log_scalars


def log_epoch_for_tensorboard(train_epoch_results, eval_epoch_results,
                              epoch, config, output_dir, output_artifacts):

    # Create the tensorboard subdirectory for this repeat (submodel of inference)
    output_artifacts['epoch_level']['tb_dir'] = os.path.join(output_dir, 'tensorboard')
    try:
        os.makedirs(output_artifacts['epoch_level']['tb_dir'], exist_ok=True)
    except Exception as e:
        logger.warning('Problem creating Tensorboard artifact directory to "{}",'
                      '\nerror={}'.format(output_artifacts['epoch_level']['tb_dir'], e))

    # Create the Tensorboard writer if it does not yet exist (on epoch 0) on the artifact directory
    if 'tb_writer' not in list(output_artifacts['epoch_level'].keys()):
        logger.info('Logging Tensorboard epoch-by-epoch results to dir = "{}"'.
                    format(output_artifacts['epoch_level']['tb_dir']))
        output_artifacts['epoch_level']['tb_writer'] = SummaryWriter(log_dir=output_artifacts['epoch_level']['tb_dir'])

    # Log training split (these are the batch-level metrics)
    output_artifacts = log_epoch_tb_metrics(results_dict=train_epoch_results,
                                            out_dir=output_artifacts['epoch_level']['tb_dir'],
                                            output_artifacts=output_artifacts,
                                            epoch=epoch,
                                            config=config,
                                            metric_type='train')

    output_artifacts = log_epoch_tb_metrics(results_dict=eval_epoch_results,
                                            out_dir=output_artifacts['epoch_level']['tb_dir'],
                                            output_artifacts=output_artifacts,
                                            epoch=epoch,
                                            config=config,
                                            metric_type='evaluation')

    return output_artifacts


def log_epoch_tb_metrics(results_dict: dict,
                         out_dir: str,
                         output_artifacts: dict,
                         epoch: int,
                         config: dict,
                         metric_type: str = 'train'):

    if metric_type == 'train':
        split_name = 'TRAINING'
        output_artifacts = log_epoch_tb_per_split(split_dict=results_dict,
                                                 split_name=split_name,
                                                 metric_type=metric_type,
                                                 epoch=epoch,
                                                 config=config,
                                                 out_dir=out_dir,
                                                 output_artifacts=output_artifacts)
    elif metric_type == 'evaluation':
        for split_name in results_dict.keys():
            output_artifacts = log_epoch_tb_per_split(split_dict=results_dict[split_name],
                                                      split_name=split_name,
                                                      metric_type=metric_type,
                                                      epoch=epoch,
                                                      config=config,
                                                      out_dir=out_dir,
                                                      output_artifacts=output_artifacts)
    else:
        raise IOError('Unknown metric_type={}!, should be either "train" or "evaluation"'.format(metric_type))

    return output_artifacts


def log_epoch_tb_per_split(split_dict: dict,
                           split_name: str,
                           metric_type: str,
                           epoch: int,
                           config: dict,
                           out_dir: str,
                           output_artifacts: dict):

    # You could have multiple datasets for test split for example, train on MINIVESS and evaluate how this
    # training generalizes to external datasets
    dataset_names = list(split_dict.keys())

    for dataset_name in dataset_names:
        output_artifacts = log_epoch_per_split_per_database(dataset_dict=split_dict[dataset_name],
                                                            split_name=split_name,
                                                            metric_type=metric_type,
                                                            dataset_name=dataset_name,
                                                            epoch=epoch,
                                                            config=config,
                                                            out_dir=out_dir,
                                                            output_artifacts=output_artifacts)
    
    return output_artifacts


def log_epoch_per_split_per_database(dataset_dict: dict,
                                     split_name: str,
                                     metric_type: str,
                                     dataset_name: str,
                                     epoch: int,
                                     config: dict,
                                     out_dir: str,
                                     output_artifacts: dict):

    variable_types = list(dataset_dict)
    for var_type in variable_types:
        if var_type == 'scalars':
            # e.g. loss per epoch
            output_artifacts = log_scalars(metric_dict=dataset_dict[var_type],
                                           var_type=var_type,
                                           split_name=split_name,
                                           metric_type=metric_type,
                                           dataset_name=dataset_name,
                                           epoch=epoch,
                                           config=config,
                                           out_dir=out_dir,
                                           output_artifacts=output_artifacts)
        elif var_type == 'arrays':
            # e.g. losses per batch
            output_artifacts = log_scalars(metric_dict=dataset_dict[var_type],
                                           var_type=var_type,
                                           split_name=split_name,
                                           metric_type=metric_type,
                                           dataset_name=dataset_name,
                                           epoch=epoch,
                                           config=config,
                                           out_dir=out_dir,
                                           output_artifacts=output_artifacts,
                                           multiple_values_per_epoch=True)
        elif var_type == 'metadata_scalars':
            # e.g. learning rate for epoch, time taken to compute one epoch, etc.
            output_artifacts = log_scalars(metric_dict=dataset_dict[var_type],
                                           var_type=var_type,
                                           split_name=split_name,
                                           metric_type=metric_type,
                                           dataset_name=dataset_name,
                                           epoch=epoch,
                                           config=config,
                                           out_dir=out_dir,
                                           output_artifacts=output_artifacts)
        else:
            raise NotImplementedError('Unknown variable_type="{}", a typo or you added some new type in addition to'
                                      'the supported "scalars", "arrays" and "metadata_scalars"')
        
    return output_artifacts