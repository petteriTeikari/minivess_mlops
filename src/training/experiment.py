from loguru import logger
from omegaconf import DictConfig

from src.utils.config_utils import import_config, cfg_key
from src.utils.dataloader_utils import define_dataset_and_dataloader
from src.training.train import training_script
from src.utils.data_utils import import_datasets


def train_run_per_hyperparameters(args: dict, log_level: str = "DEBUG"):
    # Import the config, and cfg['run'] contains derived params of this training
    cfg = import_config(
        args=args, task_cfg_name=args["task_config_file"], log_level=log_level
    )

    run_results = run_train_experiment(
        cfg=cfg, hyperparam_name=cfg_key(cfg, "run", "PARAMS", "hyperparam_name")
    )

    run_results["cfg"] = cfg

    return run_results


def define_experiment_data(cfg: dict):
    # Collect the data and define splits
    fold_split_file_dicts, cfg["run"] = import_datasets(
        data_cfg=cfg_key(cfg, "hydra_cfg", "config", "DATA"),
        cfg=cfg,
        data_dir=cfg_key(cfg, "run", "ARGS", "data_dir"),
        run_mode=cfg_key(cfg, "run", "ARGS", "run_mode"),
    )

    # Create and validate datasets and dataloaders
    if not cfg_key(cfg, "hydra_cfg", "config", "DATA", "DATALOADER", "SKIP_DATALOADER"):
        experim_datasets, experim_dataloaders = define_dataset_and_dataloader(
            cfg=cfg, fold_split_file_dicts=fold_split_file_dicts
        )
    else:
        # If you are running Github Action for checking Dataset integrity
        logger.warning(
            'Skip dataloader creation, run_mode = "{}"'.format(
                cfg_key(cfg, "run", "ARGS", "run_mode")
            )
        )
        experim_datasets, experim_dataloaders = None, None

    return fold_split_file_dicts, experim_datasets, experim_dataloaders, cfg["run"]


def run_train_experiment(cfg: dict, hyperparam_name: str):
    # Define the used data
    (
        fold_split_file_dicts,
        experim_datasets,
        experim_dataloaders,
        cfg["run"],
    ) = define_experiment_data(cfg=cfg)

    # Train for n folds, n repeats, n epochs (single model)
    if cfg_key(cfg, "hydra_cfg", "config", "TRAINING", "SKIP_TRAINING"):
        logger.info(
            "Skipping the training for now, hyperparam_name = {}".format(
                hyperparam_name
            )
        )
        run_results = {}
    else:
        run_results = training_script(
            hyperparam_name=hyperparam_name,
            experim_dataloaders=experim_dataloaders,
            cfg=cfg,
            machine_config=cfg_key(cfg, "run", "MACHINE"),
            output_dir=cfg_key(cfg, "run", "PARAMS", "output_experiment_dir"),
        )
        logger.info("Done with the experiment!")

    return run_results
