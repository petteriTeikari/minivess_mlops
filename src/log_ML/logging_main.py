import mlflow
from loguru import logger

from src.inference.ensemble_main import reinference_dataloaders
from src.log_ML.log_crossval import log_cv_results
from src.log_ML.log_epochs import log_epoch_for_tensorboard
from src.log_ML.results_utils import (
    average_repeat_results,
    reorder_crossvalidation_results,
    compute_crossval_stats,
    reorder_ensemble_crossvalidation_results,
    compute_crossval_ensemble_stats,
    get_best_repeat_result,
)
from src.log_ML.s3_utils import sync_artifacts_to_s3
from src.log_ML.wandb_log.wandb_log import (
    log_wandb_repeat_results,
    log_wandb_ensemble_results,
)
from src.utils.dict_utils import cfg_key


def log_epoch_results(
    train_epoch_results, eval_epoch_results, epoch, cfg, output_dir, output_artifacts
):
    if "epoch_level" not in list(output_artifacts.keys()):
        output_artifacts["epoch_level"] = {}

    output_artifacts = log_epoch_for_tensorboard(
        train_epoch_results,
        eval_epoch_results,
        epoch,
        cfg,
        output_dir,
        output_artifacts,
    )

    return output_artifacts


def log_n_epochs_results(
    train_results,
    eval_results,
    best_dict,
    output_artifacts,
    cfg,
    repeat_idx: int,
    fold_name: str,
    repeat_name: str,
):
    logger.debug(
        "Placeholder for n epochs logging (i.e. submodel or single repeat training)"
    )


def log_averaged_and_best_repeats(
    repeat_results: dict, fold_name: str, cfg: dict, dataloaders: dict, device
):
    # Average the repeat results (not ensembling per se yet, as we are averaging the metrics here, and not averaging
    # the predictions and then computing the metrics from the averaged predictions)
    averaged_results = average_repeat_results(repeat_results)

    # You might want to compare single model performance to ensemble and quantify the improvement from
    # added computational cost from ensembling over single repeat (submodel)
    best_repeat_dicts = get_best_repeat_result(repeat_results)

    # These are not defined for WANDB at this point, but written at the end of the script
    service = (
        "MLflow"
        if cfg_key(
            cfg, "hydra_cfg", "config", "LOGGING", "MLFLOW", "TRACKING", "enable"
        )
        else None
    )

    if cfg_key(cfg, "hydra_cfg", "config", "VALIDATION_BEST", "enable"):
        # Re-inference the dataloader with the best model(s) of the best repeat
        # e.g. you want to have Hausdorff distance here that you thought to be too heavy to compute while training
        artifacts_output_dir = cfg_key(cfg, "run", "PARAMS", "output_experiment_dir")
        best_repeat_metrics = reinference_dataloaders(
            input_dict=best_repeat_dicts,
            cfg=cfg,
            artifacts_output_dir=artifacts_output_dir,
            dataloaders=dataloaders,
            device=device,
            model_scheme="best_repeats",
        )

        # log best repeat metrics here to MLflow/WANDB
        log_best_reinference_metrics(
            best_repeat_metrics=best_repeat_metrics,
            cfg=cfg,
            fold_name=fold_name,
            service=service,
        )

    else:
        logger.info(
            'Skip "VALIDATION_BEST", no re-computation of "heavier metrics", '
            "just logging the ones obtained during training"
        )

        # Log the metric results of the best repeat out of n repeats
        log_best_repeats(
            best_repeat_dicts=best_repeat_dicts,
            cfg=cfg,
            service=service,
            fold_name=fold_name,
        )


def log_best_repeats(
    best_repeat_dicts: dict,
    cfg: dict,
    splits: tuple = ("VAL", "TEST"),
    service: str = "MLflow",
    fold_name: str = None,
):
    metric_cfg = cfg_key(cfg, "hydra_cfg", "config", "LOGGING", "MAIN_METRIC")

    if service is not None:
        logger.info("Logging (MLflow) the metrics obtained from best repeat")
        for dataset_train in best_repeat_dicts:
            for tracked_metric in best_repeat_dicts[dataset_train]:
                for split in best_repeat_dicts[dataset_train][tracked_metric]:
                    if split in splits:
                        split_dict = best_repeat_dicts[dataset_train][tracked_metric][
                            split
                        ]
                        for dataset_eval in split_dict:
                            for metric in split_dict[dataset_eval]:
                                best_repeat = split_dict[dataset_eval][metric]
                                metric_name = "{}/bestRepeat_{}_{}/{}/{}/{}".format(
                                    fold_name,
                                    metric,
                                    split,
                                    dataset_train,
                                    tracked_metric,
                                    dataset_eval,
                                )
                                metric_value = best_repeat["best_value"]
                                logger.info(
                                    '{} | "{}": {:.3f}'.format(
                                        service, metric_name, metric_value
                                    )
                                )

                                if service == "MLflow":
                                    try:
                                        mlflow.log_metric(metric_name, metric_value)
                                    except Exception as e:
                                        raise IOError(
                                            "Problem with the MLflow logging, e = {},\n"
                                            'Why one gets this "repo not associated with run" '
                                            "a bit stochastically "
                                            "and not every time?".format(e)
                                        )
                                else:
                                    raise NotImplementedError(
                                        'Unknown Experiment Tracking service = "{}"'.format(
                                            service
                                        )
                                    )

                                metric_main = correct_key_for_main_result(
                                    metric_name=metric_name,
                                    fold_name=fold_name,
                                    tracked_metric=tracked_metric,
                                    metric=metric,
                                    dataset=dataset_eval,
                                    split=split,
                                    metric_cfg=metric_cfg,
                                )

                                if metric_main != metric_name:
                                    if service == "MLflow":
                                        mlflow.log_metric(metric_main, metric_value)
                                        logger.info(
                                            '{} (main) | "{}": {:.3f}'.format(
                                                service, metric_main, metric_value
                                            )
                                        )

    else:
        logger.info(
            "Skipping the logging of the best repeat results, "
            "as no Experiment tracking service enabled (namely MLflow at this point)"
        )


def log_crossvalidation_results(
    fold_results: dict,
    ensembled_results: dict,
    experim_dataloaders: dict,
    cfg: dict,
    output_dir: str,
):
    # Reorder CV results and compute the stats, i.e. mean of 5 folds and 5 repeats per fold (n = 25)
    # by default, computes for all the splits, but you most likely are the most interested in the TEST
    fold_results_reordered = reorder_crossvalidation_results(fold_results)
    cv_results = compute_crossval_stats(fold_results_reordered)

    ensembled_results_reordered = reorder_ensemble_crossvalidation_results(
        ensembled_results
    )
    if ensembled_results_reordered is not None:
        cv_ensemble_results = compute_crossval_ensemble_stats(
            ensembled_results_reordered
        )
        # sample_cv_results = get_cv_sample_stats_from_ensemble(ensembled_results)
    else:
        cv_ensemble_results = None

    if cfg_key(cfg, "hydra_cfg", "config", "LOGGING", "WANDB", "enable"):
        log_wandb_repeat_results(
            fold_results=fold_results,
            output_dir=cfg_key(cfg, "run", "PARAMS", "output_base_dir"),
            cfg=cfg,
        )
    else:
        logger.info("Skipping repeat-level WANDB Logging!")

    if cv_ensemble_results is not None:
        if cfg_key(cfg, "hydra_cfg", "config", "LOGGING", "WANDB", "enable"):
            log_wandb_ensemble_results(
                ensembled_results=ensembled_results,
                output_dir=cfg_key(cfg, "run", "PARAMS", "output_base_dir"),
                cfg=cfg,
            )

        cv_averaged_output_dir = cfg_key(
            cfg, "run", "PARAMS", "cross_validation_averaged"
        )
        cv_ensembled_output_dir = cfg_key(
            cfg, "run", "PARAMS", "cross_validation_ensembled"
        )
        output_dir = cfg["run"]["PARAMS"]["output_experiment_dir"]
        logged_model_paths = log_cv_results(
            cv_results=cv_results,
            cv_ensemble_results=cv_ensemble_results,
            ensembled_results=ensembled_results,
            fold_results=fold_results,
            experim_dataloaders=experim_dataloaders,
            cfg=cfg,
            output_dir=output_dir,
            cv_averaged_output_dir=cv_averaged_output_dir,
            cv_ensembled_output_dir=cv_ensembled_output_dir,
        )
    else:
        logged_model_paths = None

    if mlflow.active_run() is not None:
        mlflow.end_run()
        logger.info("Done with the MLflow Logging!")

    sync_artifacts_to_s3(experiment_artifacts_dir=output_dir, cfg=cfg)

    return logged_model_paths


def log_best_reinference_metrics(
    best_repeat_metrics: dict,
    cfg: dict,
    stat_key: str = "split_metrics_stat",
    stat_value_to_log: str = "mean",
    service: str = "MLflow",
    fold_name: str = None,
):
    for split in list(best_repeat_metrics.keys()):
        split_stats = best_repeat_metrics[split][stat_key]
        for dset1 in split_stats:
            for tracked_metric in split_stats[dset1]:
                for dset2 in split_stats[dset1][tracked_metric]:
                    metrics = split_stats[dset1][tracked_metric][dset2]["metrics"]
                    for metric in metrics:
                        stats_dict = metrics[metric]
                        metric_name = "{}/bestRepeat_{}_{}/{}/{}/{}".format(
                            fold_name, metric, split, dset1, tracked_metric, dset2
                        )
                        value = stats_dict[stat_value_to_log]
                        logger.info(
                            '{} | "{}": {:.3f}'.format(service, metric_name, value)
                        )

                        # TO-OPTIMIZE: If you start having multiple tracked metrics, and multiple metrics,
                        # you start to have millions of columns, and you might want to define in your config
                        # "a main metric" that it gets easier to quickly check out your experiments?

                        if service == "MLflow":
                            mlflow.log_metric(metric_name, value)
                        else:
                            raise NotImplementedError(
                                'Unknown Experiment Tracking service = "{}"'.format(
                                    service
                                )
                            )

                        metric_main = correct_key_for_main_result(
                            metric_name=metric_name,
                            fold_name=fold_name,
                            tracked_metric=tracked_metric,
                            metric=metric,
                            dataset=dset2,
                            split=split,
                            metric_cfg=config["config"]["LOGGING"]["MAIN_METRIC"],
                        )

                        if metric_main != metric_name:
                            if service == "MLflow":
                                mlflow.log_metric(metric_main, value)
                                logger.info(
                                    '{} (main) | "{}": {:.3f}'.format(
                                        service, metric_main, value
                                    )
                                )


def correct_key_for_main_result(
    metric_name: str,
    fold_name: str,
    tracked_metric: str,
    metric: str,
    split: str,
    dataset: str,
    metric_cfg: dict,
):
    if (
        tracked_metric == metric_cfg["tracked_metric"]
        and metric == metric_cfg["metric"]
        and dataset == metric_cfg["dataset"]
    ):
        metric_type = metric_name.split("/")[1].split("_")[0]
        metric_name = fold_name + "/" + metric_type + "_" + split

    return metric_name
