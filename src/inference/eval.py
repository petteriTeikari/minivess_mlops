import time

import numpy as np
import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose

from src.utils.train_utils import init_epoch_dict, get_timings_per_epoch


def evaluate_datasets_per_epoch(
    model, device, epoch, dataloaders, training_config, metric_dict, split_name
):
    epoch_metrics = init_epoch_dict(
        epoch, dataloaders[split_name], split_name=split_name
    )
    for j, dataset_name in enumerate(dataloaders[split_name].keys()):
        dataloader = dataloaders[split_name][dataset_name]
        epoch_metrics[dataset_name] = evaluate_1_epoch(
            dataloader,
            model,
            split_name,
            dataset_name,
            training_config,
            metric_dict,
            device,
            epoch_metrics[dataset_name],
        )

    return epoch_metrics


def evaluate_1_epoch(
    dataloader,
    model,
    split_name,
    dataset_name,
    training_config,
    metric_dict,
    device,
    epoch_metrics_per_dataset,
):
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # https://github.com/Project-MONAI/tutorials/blob/2183d45f48c53924b291a16d72f8f0e0b29179f2/acceleration/distributed_training/brats_training_ddp.py#L341
    model.eval()
    batch_szs = []
    epoch_start = time.time()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if training_config["PRECISION"] == "AMP":
                with torch.cuda.amp.autocast():
                    val_outputs = sliding_window_inference(
                        inputs=batch_data["image"].to(device), **metric_dict
                    )
            else:
                val_outputs = sliding_window_inference(
                    inputs=batch_data["image"].to(device), **metric_dict
                )
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

            dice_metric(y_pred=val_outputs, y=batch_data["label"].to(device))
            dice_metric_batch(y_pred=val_outputs, y=batch_data["label"].to(device))
            batch_szs.append(batch_data["image"].shape[0])

        # FIXME! Actually use the config for these to determine which are tracked
        epoch_metrics_per_dataset["scalars"]["dice"] = dice_metric.aggregate().item()
        # epoch_metrics_per_dataset['scalars']['dice_batch'] = float(dice_metric_batch.aggregate().detach().cpu())
        dice_metric.reset()
        dice_metric_batch.reset()

        epoch_metrics_per_dataset = get_timings_per_epoch(
            metadata_dict=epoch_metrics_per_dataset,
            epoch_start=epoch_start,
            no_batches=batch_idx + 1,
            mean_batch_sz=float(np.mean(batch_szs)),
        )

    return epoch_metrics_per_dataset
