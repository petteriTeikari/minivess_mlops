import numpy as np
import time
from loguru import logger

import torch
from monai.metrics import compute_hausdorff_distance, compute_generalized_dice
from monai.utils import convert_to_tensor
from monai.networks.utils import one_hot


def get_sample_metrics_from_np_masks(
    x: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    eval_config: dict,
    include_background: bool = False,
    debug_mode: bool = True,
):
    # Do the wrangling and checks
    x, y, y_pred, y_onehot, y_pred_onehot, sample_metrics = prepare_for_metrics(
        y=y, y_pred=y_pred, x=x
    )

    metrics_computed = 0

    # Hausdorff, takes a lot longer than Dice but if only evaluated once after all the repeats per sample,
    # this does not get some computationally heavy
    if "hausdorff" in eval_config["METRICS"]:
        t0 = time.time()
        metric = "hausdorff"
        sample_metrics["metrics"][metric] = (
            compute_hausdorff_distance(
                y_pred=y_pred_onehot, y=y_onehot, include_background=include_background
            )
            .detach()
            .squeeze()
            .numpy()
        )
        sample_metrics["metrics"][metric] = np.expand_dims(
            sample_metrics["metrics"][metric], axis=0
        )
        sample_metrics["timing"][metric] = np.array([time.time() - t0])
        metrics_computed += 1

    if "dice" in eval_config["METRICS"]:
        t0 = time.time()
        metric = "dice"
        sample_metrics["metrics"][metric] = (
            compute_generalized_dice(
                y_pred=y_pred_onehot, y=y_onehot, include_background=include_background
            )
            .detach()
            .numpy()
        )
        sample_metrics["timing"][metric] = np.array([time.time() - t0])
        metrics_computed += 1

    if metrics_computed == 0:
        logger.warning(
            "No metrics computed after re-inference, is your config file correct with these"
            "metrics to be computed:\n{}".format(eval_config["METRICS"])
        )

    return sample_metrics


def prepare_for_metrics(y, y_pred, x):
    def one_hot_encode(array):
        # https://stackoverflow.com/q/73451052
        return np.eye(5)[array].astype(dtype=int)

    sample_metrics = {"metrics": {}, "timing": {}}

    assert len(y.shape) == 5, "Input should be 5D (BCHWD), now it is {}D".format(
        len(y.shape)
    )

    assert (
        y.shape[0] == y_pred.shape[0]
    ), "1st dimension is not the same for pred and ground truth"
    assert (
        y.shape[1] == y_pred.shape[1]
    ), "2nd dimension is not the same for pred and ground truth"
    assert (
        y.shape[2] == y_pred.shape[2]
    ), "3rd dimension is not the same for pred and ground truth"
    y_no_classes = len(np.unique(y))

    x = convert_to_tensor(x, track_meta=True)
    y = convert_to_tensor(y, track_meta=True)  # e.g. (512,512,39) -> (1,512,512,39)
    y_pred = convert_to_tensor(y_pred, track_meta=True)

    # 'BCHW[D]' (batch, (one hot), channel, height, width, depth)
    y_onehot = one_hot(
        torch.unsqueeze(y, dim=1), num_classes=2, dim=1
    )  # (e.g. 1,2,512,512,39)
    y_pred_onehot = one_hot(torch.unsqueeze(y_pred, dim=1), num_classes=2, dim=1)

    assert (
        len(y_onehot.shape) == 6
    ), "y_onehot should be 6D (B,HotEncClass,C,H,W,D), now it is {}D".format(
        len(y_onehot)
    )

    return x, y, y_pred, y_onehot, y_pred_onehot, sample_metrics


def get_sample_uq_metrics_from_ensemble_stats(ensemble_stat_results):
    sample_metrics = {"metrics": {}, "timing": {}}

    # t0 = time.time()
    # metric = 'meanVar'  # quick'n'dirty estimate on how different are the predictions between repeats (submodels)
    # sample_metrics['metrics'][metric] = np.array([np.mean(ensemble_stat_results['arrays']['var'])])
    # sample_metrics['timing'][metric] = np.array([time.time() - t0])

    return sample_metrics
