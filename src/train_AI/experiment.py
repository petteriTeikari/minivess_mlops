from loguru import logger

from src.utils.dataloader_utils import define_dataset_and_dataloader
from src.train_AI.train import training_script
from src.utils.data_utils import import_datasets
from src.utils.metadata_utils import get_run_metadata


def run_train_experiment(config: dict,
                         hyperparam_name: str):

    # Add run/environment-dependent metadata (e.g. library versions, etc.)
    config['metadata'] = get_run_metadata()

    # Collect the data and define splits
    fold_split_file_dicts, config['config']['DATA'] = \
        import_datasets(data_config=config['config']['DATA'],
                        data_dir=config['ARGS']['data_dir'],
                        run_mode=config['ARGS']['run_mode'],
                        config=config)

    # Create and validate datasets and dataloaders
    if not config['config']['DATA']['DATALOADER']['SKIP_DATALOADER']:
        experim_datasets, experim_dataloaders = \
            define_dataset_and_dataloader(config=config,
                                          fold_split_file_dicts=fold_split_file_dicts)

        # Train for n folds, n repeats, n epochs (single model)
        if config['config']['TRAINING']['SKIP_TRAINING']:
            logger.info('Skipping the training for now, hyperparam_name = {}, '
                        'e.g. when running CI/CD tasks for data checks'.format(hyperparam_name))
            run_results = None
        else:
            run_results = \
                training_script(hyperparam_name=hyperparam_name,
                                experim_dataloaders=experim_dataloaders,
                                config=config,
                                machine_config=config['config']['MACHINE'],
                                output_dir=config['run']['output_experiment_dir'])
            logger.info('Done with the experiment!')
    else:
        logger.warning('Skip both dataloader creation and network training, '
                       'run_mode = {} (not a warning for data CI/CD)'.format(config['ARGS']['run_mode']))
        run_results = None

    return run_results
