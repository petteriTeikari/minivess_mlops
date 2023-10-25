from loguru import logger
from omegaconf import DictConfig

from src.utils.config_utils import import_config
from src.utils.dataloader_utils import define_dataset_and_dataloader
from src.training.train import training_script
from src.utils.data_utils import import_datasets


def train_run_per_hyperparameters(args: dict,
                                  log_level: str = 'DEBUG'):

    # Import the config, and exp_run contains derived params of this training
    config, exp_run = import_config(args=args,
                                    task_cfg_name=args['task_config_file'],
                                    log_level=log_level)

    run_results = run_train_experiment(config=config,
                                       exp_run=exp_run,
                                       hyperparam_name=exp_run['RUN']['hyperparam_name'])

    return run_results


def define_experiment_data(config: DictConfig,
                           exp_run: dict):

    # Collect the data and define splits
    fold_split_file_dicts, exp_run = \
        import_datasets(data_config=config['config']['DATA'],
                        data_dir=exp_run['ARGS']['data_dir'],
                        run_mode=exp_run['ARGS']['run_mode'],
                        exp_run=exp_run,
                        config=config)

    # Create and validate datasets and dataloaders
    if not config['config']['DATA']['DATALOADER']['SKIP_DATALOADER']:
        experim_datasets, experim_dataloaders = \
            define_dataset_and_dataloader(config=config,
                                          fold_split_file_dicts=fold_split_file_dicts,
                                          exp_run=exp_run)
    else:
        # If you are running Github Action for checking Dataset integrity
        logger.warning('Skip dataloader creation, run_mode = "{}"'.format(exp_run['ARGS']['run_mode']))
        experim_datasets, experim_dataloaders = None, None

    return fold_split_file_dicts, experim_datasets, experim_dataloaders, exp_run



def run_train_experiment(config: DictConfig,
                         exp_run: dict,
                         hyperparam_name: str):

    # Define the used data
    fold_split_file_dicts, experim_datasets, experim_dataloaders, exp_run = (
        define_experiment_data(config=config,
                               exp_run=exp_run))

    # Train for n folds, n repeats, n epochs (single model)
    if config['config']['TRAINING']['SKIP_TRAINING']:
        logger.info('Skipping the training for now, hyperparam_name = {}'.format(hyperparam_name))
        run_results = None
    else:
        run_results = \
            training_script(hyperparam_name=hyperparam_name,
                            experim_dataloaders=experim_dataloaders,
                            config=config,
                            exp_run=exp_run,
                            machine_config=exp_run['MACHINE'],
                            output_dir=exp_run['RUN']['output_experiment_dir'])
        logger.info('Done with the experiment!')

    return run_results
