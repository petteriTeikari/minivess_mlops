import os
import time
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm
from loguru import logger

from ml_tests.model_tests import model_tests_main
from src.inference.ensemble_main import reinference_dataloaders
from src.eval import evaluate_datasets_per_epoch
from src.inference.ensemble_utils import get_submodel_name
from src.log_ML.log_ensemble import log_ensemble_results
from src.log_ML.logging_main import log_epoch_results, log_n_epochs_results, \
    log_crossvalidation_results, log_averaged_and_best_repeats
from src.log_ML.model_saving import save_models_if_improved
from src.utils.model_utils import import_segmentation_model
from src.utils.train_utils import init_epoch_dict, collect_epoch_results, set_model_training_params, \
    get_timings_per_epoch


def training_script(hyperparam_name: str,
                    experim_dataloaders: dict,
                    config: dict,
                    machine_config: dict,
                    output_dir: str) -> dict:

    logger.info('Starting training for the hyperparameter config "{}"'.format(hyperparam_name))

    # Cross-validation loop (if used), i.e. when train/val splits change for each execution
    os.makedirs(output_dir, exist_ok=True)
    fold_results, ensembled_results = {}, {}
    config['run']['cross_validation'] = os.path.join(output_dir, 'cross_validation')
    config['run']['cross_validation_averaged'] = os.path.join(config['run']['cross_validation'], 'averaged')
    config['run']['cross_validation_ensembled'] = os.path.join(config['run']['cross_validation'], 'ensembled')

    for f, fold_name in enumerate(list(experim_dataloaders.keys())):
        logger.info('Training fold #{}/{}: {}'.format(f+1, len(experim_dataloaders.keys()), fold_name))
        config['run']['fold_dir'][fold_name] = os.path.join(output_dir, fold_name)
        # config['run']['repeat_artifacts'][fold_name] = os.path.join(config['run']['fold_dir'][fold_name], 'repeats')
        config['run']['ensemble_artifacts'][fold_name] = os.path.join(config['run']['fold_dir'][fold_name], 'ensemble')

        # Quick'n'dirty placeholder to simulate diverse ensembles, i.e. when your submodels are not just
        # different repeats (as in different random starting points) but different model architectures, e.g.
        # 1) CNN, 2) Transformer, etc. (and each of these individual architectures can be repeated n times)
        fold_results[fold_name], ensembled_results[fold_name] = \
            train_model_for_single_fold(fold_dataloaders=experim_dataloaders[fold_name],
                                        config=config,
                                        # Note! these are now parsed from input .yaml
                                        hyperparameters_config=config['hyperparameters'],
                                        machine_config=machine_config,
                                        output_dir=config['run']['fold_dir'][fold_name],
                                        fold_name=fold_name)

    logger.info('Done training all the {} folds'.format(len(experim_dataloaders)))

    config['run']['logged_model_paths'] = (
        log_crossvalidation_results(fold_results=fold_results,
                                    ensembled_results=ensembled_results,
                                    experim_dataloaders=experim_dataloaders,
                                    config=config,
                                    output_dir=output_dir))

    logger.info('Done training the hyperparameter config "{}"'.format(hyperparam_name))

    return {'fold_results':fold_results, 'ensembled_results': ensembled_results, 'config': config}


def train_model_for_single_fold(fold_dataloaders: dict,
                                config: dict,
                                hyperparameters_config: dict,
                                machine_config: dict,
                                output_dir: str,
                                fold_name: str):
    """
    Up to you decide if you consider what is really a different architecture within the ensemble
    """
    architecture_names = list(hyperparameters_config['models'].keys())
    output_dir = os.path.join(output_dir, 'submodels')
    os.makedirs(output_dir, exist_ok=True)

    archi_results = {}
    for i, architecture_name in enumerate(architecture_names):
        logger.info('Training architecture #{}/{}: {}'.format(i+1, len(architecture_names), architecture_name))
        # Pick the proper architecture and training params per architecture so that
        # "train_model_for_single_architecture()" sees just the "normal model" to be trained for n repeats
        training_config = hyperparameters_config['models'][architecture_name]['training']
        model_config = hyperparameters_config['models'][architecture_name]['architecture']

        # TODO! Re-define dataloader on each iteration of the different architecture
        #  e..g. if you want to do diverse ensembles with each submodel with different augmentation
        archi_results[architecture_name] = \
            train_model_for_single_architecture(fold_dataloaders=fold_dataloaders,
                                                config=config,
                                                training_config=training_config,
                                                model_config=model_config,
                                                machine_config=machine_config,
                                                output_dir=output_dir,
                                                archi_name=architecture_name,
                                                fold_name=fold_name)

    # Ensemble the repeats (submodels)
    if config['config']['ENSEMBLE']['enable']:
        ensemble_results = reinference_dataloaders(input_dict=archi_results,
                                                   dataloaders=fold_dataloaders,
                                                   artifacts_output_dir=config['run']['ensemble_artifacts'][
                                                       fold_name],
                                                   config=config,
                                                   device=machine_config['IN_USE']['device'],
                                                   model_scheme='ensemble_from_repeats')
        # Log inference
        log_ensemble_results(ensemble_results, config=config, fold_name=fold_name)

    else:
        logger.info('Skip ENSEMBLING')
        ensemble_results = None

    return archi_results, ensemble_results


def train_model_for_single_architecture(fold_dataloaders: dict,
                                        config: dict,
                                        training_config: dict,
                                        model_config: dict,
                                        machine_config: dict,
                                        output_dir: str,
                                        archi_name: str,
                                        fold_name: str) -> dict:

    # Repeat n times the same data fold (i.e. you get n submodels for an inference)
    os.makedirs(output_dir, exist_ok=True)

    repeat_results = {}
    for repeat_idx in range(training_config['NO_REPEATS']):
        logger.info('Training repeat #{}/{}'.format(repeat_idx + 1, training_config['NO_REPEATS']))
        repeat_name = 'repeat{}'.format(str(repeat_idx+1).zfill(2))
        submodel_name = get_submodel_name(repeat_name=repeat_name, archi_name=archi_name)
        repeat_results[repeat_name] = \
            train_single_model(dataloaders=fold_dataloaders,
                               config=config,
                               training_config=training_config,
                               model_config=model_config,
                               machine_config=machine_config,
                               repeat_idx=repeat_idx,
                               repeat_name=repeat_name,
                               device=machine_config['IN_USE']['device'],
                               # output_dir=os.path.join(output_dir, archi_name, repeat_name),
                               archi_name=archi_name,
                               output_dir=os.path.join(output_dir, submodel_name),
                               fold_name=fold_name)

    logger.info('Done training all the {} repeats of "{}"'.format(training_config['NO_REPEATS'], fold_name))

    # Log (repeat averages) and best repeats
    log_averaged_and_best_repeats(repeat_results,
                                  fold_name=fold_name,
                                  config=config,
                                  dataloaders=fold_dataloaders,
                                  device=machine_config['IN_USE']['device'])

    return repeat_results


def train_single_model(dataloaders: dict,
                       config: dict,
                       training_config: dict,
                       model_config: dict,
                       machine_config: dict,
                       repeat_idx: int,
                       repeat_name: str,
                       archi_name: str,
                       fold_name: str,
                       device,
                       output_dir: str) -> dict:

    os.makedirs(output_dir, exist_ok=True)
    if training_config['PRECISION'] == 'AMP':
        scaler = torch.cuda.amp.GradScaler()
    else:
        raise NotImplementedError('Check the train loop also for non-AMP operation')

    # Define the model to be used
    model = import_segmentation_model(model_config, archi_name, device)

    # Model training params
    # TODO! if you some LDAM-DRW type of scheme for class-imbalanced learning, you might re-define loss
    #  on each epoch as it is the only way to change to class weights dynamically?
    loss_function, optimizer, lr_scheduler = \
        set_model_training_params(model, device, scaler, training_config, config)

    # Train script
    train_results, eval_results, best_dict, output_artifacts = \
        train_n_epochs_script(model, dataloaders,
                              device, scaler,
                              loss_function, optimizer, lr_scheduler,
                              training_config, config, output_dir=output_dir,
                              repeat_idx=repeat_idx, fold_name=fold_name, repeat_name=repeat_name)

    # Post-train scripts (if done)
    post_train_n_epochs_script()

    # When training is done, you con for example log the repeat/experiment/n_epochs level results
    log_n_epochs_results(train_results, eval_results, best_dict, output_artifacts, config,
                         repeat_idx=repeat_idx, fold_name=fold_name, repeat_name=repeat_name)



    results_out = {
                   'train_results': train_results,
                   'eval_results': eval_results,
                   'best_dict': best_dict,
                   'output_artifacts': output_artifacts,
                   }

    logger.info('Done training "{}" of "{}"'.format(repeat_name, fold_name))

    return results_out


def train_n_epochs_script(model, dataloaders,
                          device, scaler,
                          loss_function, optimizer, lr_scheduler,
                          training_config: dict, config: dict,
                          start_epoch: int = 0, output_dir: str = None,
                          repeat_idx: int = None, fold_name: str = None, repeat_name: str = None):

    # FIXME: get this from config
    metric_dict = {'roi_size': (64, 64, 8), 'sw_batch_size': 4, 'predictor': model, 'overlap': 0.6}

    train_results = {}
    eval_results = {}
    output_artifacts = {'output_dir': output_dir}
    best_dicts = None

    # https://github.com/Project-MONAI/tutorials/blob/2183d45f48c53924b291a16d72f8f0e0b29179f2/acceleration/distributed_training/brats_training_ddp.py#L285
    print(' ')
    for epoch in tqdm(range(start_epoch, training_config['NUM_EPOCHS']),
                      desc='Training the network, {}, repeat {}, epoch#'.format(fold_name, repeat_idx+1),
                      position=0):

        eval_epoch_results = {}

        init_model = deepcopy(model) # for ML tests, track if model has changed
        if epoch == start_epoch:
            model_test_metrics0, model_tests0 = (
                model_tests_main(model,
                                 initial_model=None,
                                 test_config=config['config']['TESTING']['MODEL'],
                                 first_epoch=True))

        # Train
        train_epoch_results, eval_epoch_results['TRAIN'] = \
            train_1_epoch(model, device, epoch, loss_function, optimizer, lr_scheduler, scaler, training_config,
                          train_loader=dataloaders['TRAIN'], metric_dict=metric_dict,
                          dataset_dummy_key='MINIVESS')

        # Validate (as in decide whether the model improved or not)
        split_name = 'VAL'
        eval_epoch_results[split_name] = evaluate_datasets_per_epoch(model, device, epoch, dataloaders,
                                                                     training_config, metric_dict, split_name)

        # Evaluate (check generalization for held-out datasets)
        split_name = 'TEST'
        eval_epoch_results[split_name] = evaluate_datasets_per_epoch(model, device, epoch, dataloaders,
                                                                     training_config, metric_dict, split_name)

        # ML Tests (again)
        model_test_metrics, model_tests = (
            model_tests_main(model,
                             initial_model=None,
                             test_config=config['config']['TESTING']['MODEL'],
                             first_epoch=True))

        # Collect results to a dictionary and avoid having multiple lists for each metric
        train_results, eval_results = collect_epoch_results(train_epoch_results, eval_epoch_results,
                                                            train_results, eval_results, epoch)

        # Log epoch-level result
        output_artifacts = log_epoch_results(train_epoch_results, eval_epoch_results,
                                             epoch, config, output_dir, output_artifacts)

        # Save model(s) if model has improved
        output_artifacts['model_dir'] = os.path.join(output_dir, 'models')
        best_dicts = save_models_if_improved(best_dicts, epoch,
                                             model, optimizer, lr_scheduler,
                                             train_epoch_results, train_results,
                                             eval_epoch_results, eval_results,
                                             validation_config=config['config']['VALIDATION'],
                                             config=config,
                                             model_dir=output_artifacts['model_dir'],
                                             fold_name=fold_name, repeat_name=repeat_name)

    return train_results, eval_results, best_dicts, output_artifacts


def train_1_epoch(model, device, epoch, loss_function, optimizer, lr_scheduler, scaler, training_config,
                 train_loader, metric_dict, dataset_dummy_key: str = 'MINIVESS'):

    # https://github.com/Project-MONAI/tutorials/blob/2183d45f48c53924b291a16d72f8f0e0b29179f2/acceleration/distributed_training/brats_training_ddp.py#L317
    model.train()
    epoch_eval_res = init_epoch_dict(epoch, loaders = {dataset_dummy_key: train_loader}, split_name='TRAIN')
    epoch_trn_res = init_epoch_dict(epoch, loaders = {dataset_dummy_key: train_loader}, split_name='TRAIN')
    batch_losses = []
    batch_szs = []

    epoch_start = time.time()
    # tqdm(x, position=1, leave=False)  # this would be an inner loop inside the outer loop of epoch tqdm
    for batch_idx, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        loss = train_1_batch(model, device, batch_data, loss_function,
                             amp_on=training_config['PRECISION']=='AMP')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_losses.append(loss.item())
        batch_szs.append(batch_data['image'].shape[0])

    lr_scheduler.step()

    # Average loss per epoch (no_of_steps = batch_idx+1)
    epoch_eval_res[dataset_dummy_key]['scalars']['loss'] = np.mean(batch_losses)

    # Batch-level losses (depending on what you want to see, maybe both on MLflow/Wandb/etc.)
    epoch_trn_res[dataset_dummy_key]['arrays']['batch_loss'] = np.array(batch_losses)

    # You could dump metadata here about the actual training process (that is not autocaptured bu WandB for example)
    epoch_trn_res[dataset_dummy_key]['metadata_scalars']['lr'] = lr_scheduler.get_last_lr()[0]
    epoch_trn_res[dataset_dummy_key] = get_timings_per_epoch(metadata_dict=epoch_trn_res[dataset_dummy_key],
                                                             epoch_start=epoch_start,
                                                             no_batches=batch_idx+1,
                                                             mean_batch_sz=float(np.mean(batch_szs)))

    return epoch_trn_res, epoch_eval_res


def train_1_batch(model, device, batch_data, loss_function, amp_on: bool = True):

    if amp_on:
        with torch.cuda.amp.autocast():
            outputs = model(batch_data["image"].to(device)) # .to(device) here instead of "ToDeviced"
            # "ToDeviced" lead to GPU memory glitches, inspect this later?
            loss = loss_function(outputs, batch_data["label"].to(device)) # .to(device) here instead of "ToDeviced"
    else:
        outputs = model(batch_data["image"].to(device))  # .to(device) here instead of "ToDeviced"
        # "ToDeviced" lead to GPU memory glitches, inspect this later?
        loss = loss_function(outputs, batch_data["label"].to(device))  # .to(device) here instead of "ToDeviced"

    return loss


def post_train_n_epochs_script():

    # TODO! Any post-repeat training stuff could happen here, any plug-n-play methods to improve the model
    #  SWA, Multi-SWAG, Last Layer Re-Training [Polina Kirichenko et al. (2022)], etc.
    logger.debug('Placeholder for any post n epochs training')