import torch
import torch.distributed as dist
from tqdm import tqdm

from src.utils.general_utils import check_if_key_in_dict
from src.utils.model_utils import import_segmentation_model

from monai.data import ThreadDataLoader, partition_dataset, decollate_batch
from monai.optimizers import Novograd
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose
)

def train_model(dataloaders,
                config: dict,
                training_config: dict,
                model_config: dict,
                machine_config: dict,
                local_rank: int = 0):

    def choose_loss_function(training_config: dict, loss_config: dict):
        loss_name = training_config['LOSS']['NAME']
        if check_if_key_in_dict(loss_config, loss_name):
            # FIXME parse names to match any MONAI loss
            loss_params = loss_config[loss_name] # FIXME autouse these in the function call
            if loss_name == 'DiceFocalLoss':
                loss_function = DiceFocalLoss(
                    smooth_nr=1e-5,
                    smooth_dr=1e-5,
                    squared_pred=True,
                    to_onehot_y=False,
                    sigmoid=True,
                    batch=True,
                )
            else:
                raise NotImplementedError('Unsupported loss_name = "{}"'.format(loss_name))
        else:
            raise IOError('Could not find loss config for loss_name = "{}"'.format(loss_name))

        return loss_function


    def choose_optimizer(model, training_config: dict, optimizer_config: dict):

        optimizer_name = training_config['OPTIMIZER']['NAME']
        if check_if_key_in_dict(optimizer_config, optimizer_name):
            # FIXME parse names to match any MONAI/PyTorch optimizer
            optimizer_params = optimizer_config[optimizer_name]  # FIXME autouse these in the function call
            if optimizer_name == 'Novograd':
                optimizer = Novograd(model.parameters(), lr=training_config['LR'])
            else:
                raise NotImplementedError('Unsupported optimizer_name = "{}"'.format(optimizer_name))
        else:
            raise IOError('Could not find optimizer config for optimizer_name = "{}"'.format(optimizer_name))

        return optimizer


    def choose_lr_scheduler(optimizer, training_config: dict, scheduler_config: dict):

        scheduler_name = training_config['SCHEDULER']['NAME']
        if check_if_key_in_dict(scheduler_config, scheduler_name):
            # FIXME parse names to match any MONAI/Pytorch Scheduler
            scheduler_params = scheduler_config[scheduler_name]  # FIXME autouse these in the function call
            if scheduler_name == 'CosineAnnealingLR':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                          T_max=training_config['NUM_EPOCHS'])
            else:
                raise NotImplementedError('Unsupported scheduler_name = "{}"'.format(scheduler_name))
        else:
            raise IOError('Could not find optimizer config for scheduler_name = "{}"'.format(scheduler_name))

        return lr_scheduler


    def init_training(machine_config: dict, training_config: dict, config: dict, local_rank: int = 0):

        if machine_config['DISTRIBUTED']:
            # initialize the distributed training process, every GPU runs in a process
            # see e.g.
            # https://github.com/Project-MONAI/tutorials/blob/main/acceleration/fast_model_training_guide.md
            # https://github.com/Project-MONAI/tutorials/blob/main/acceleration/distributed_training/brats_training_ddp.py
            dist.init_process_group(backend="nccl", init_method="env://")

        device = torch.device(f"cuda:{local_rank}") # FIXME! allow CPU-training for devel/debugging purposes as well
        torch.cuda.set_device(device)
        if training_config['AMP']:
            scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True

        return device, scaler


    def set_model_training_params(model, device, scaler, training_config: dict, config: dict):

        loss_function = choose_loss_function(training_config = training_config,
                                             loss_config = config['config']['LOSS_FUNCTIONS'])

        optimizer = choose_optimizer(model = model,
                                     training_config = training_config,
                                     optimizer_config = config['config']['OPTIMIZERS'])

        lr_scheduler = choose_lr_scheduler(optimizer = optimizer,
                                           training_config=training_config,
                                           scheduler_config=config['config']['SCHEDULERS'])

        return loss_function, optimizer, lr_scheduler


    # Do the init stuff
    device, scaler,  = \
        init_training(machine_config=machine_config,
                      training_config=training_config,
                      config=config,
                      local_rank=local_rank)

    # Define the model to be used
    model = import_segmentation_model(model_config, device)

    # Model training params
    loss_function, optimizer, lr_scheduler = \
        set_model_training_params(model, device, scaler, training_config, config)

    # Train script
    train_n_epochs_script(model, dataloaders,
                          device, scaler,
                          loss_function, optimizer, lr_scheduler,
                          training_config, config)


def train_n_epochs_script(model, dataloaders,
                          device, scaler,
                          loss_function, optimizer, lr_scheduler,
                          training_config: dict, config: dict):

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    results_dict = {}

    # https://github.com/Project-MONAI/tutorials/blob/2183d45f48c53924b291a16d72f8f0e0b29179f2/acceleration/distributed_training/brats_training_ddp.py#L285
    for epoch in tqdm(range(training_config['NUM_EPOCHS']), desc = 'Training the network'):

        # Train
        trn_metrics = train_1_epoch(model, loss_function, optimizer, lr_scheduler, scaler,
                                    train_loader = dataloaders['TRAIN'])

        # Evaluate
        val_metrics = evaluate_1_epoch(model, dataloaders, training_config)


def train_1_epoch(model, loss_function, optimizer, lr_scheduler, scaler,
                 train_loader):

    # https://github.com/Project-MONAI/tutorials/blob/2183d45f48c53924b291a16d72f8f0e0b29179f2/acceleration/distributed_training/brats_training_ddp.py#L317
    a = 1

    return {}


def evaluate_1_epoch(model, dataloaders, training_config, eval_split_keys: tuple = ('VAL', 'TEST')):

    for i, split_name in enumerate(eval_split_keys):
        for j, dataset_name in enumerate(dataloaders[split_name].keys()):
            dataloader = dataloaders[split_name][dataset_name]
            evaluate_1_split_1_dataset(dataloader, model, split_name, dataset_name, training_config)


def evaluate_1_split_1_dataset(dataloader, model, split_name, dataset_name, training_config):

    # https://github.com/Project-MONAI/tutorials/blob/2183d45f48c53924b291a16d72f8f0e0b29179f2/acceleration/distributed_training/brats_training_ddp.py#L341
    a = 1