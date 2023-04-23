import torch
import torch.distributed as dist

from monai.losses import DiceFocalLoss
from monai.optimizers import Novograd

from src.utils.general_utils import check_if_key_in_dict


def choose_loss_function(training_config: dict, loss_config: dict):
    loss_name = training_config['LOSS']['NAME']
    if check_if_key_in_dict(loss_config, loss_name):
        # FIXME parse names to match any MONAI loss
        loss_params = loss_config[loss_name]  # FIXME autouse these in the function call
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

    device = torch.device(f"cuda:{local_rank}")  # FIXME! allow CPU-training for devel/debugging purposes as well
    torch.cuda.set_device(device)
    if training_config['AMP']:
        scaler = torch.cuda.amp.GradScaler()
    torch.backends.cudnn.benchmark = True

    return device, scaler


def set_model_training_params(model, device, scaler, training_config: dict, config: dict):
    loss_function = choose_loss_function(training_config=training_config,
                                         loss_config=config['config']['LOSS_FUNCTIONS'])

    optimizer = choose_optimizer(model=model,
                                 training_config=training_config,
                                 optimizer_config=config['config']['OPTIMIZERS'])

    lr_scheduler = choose_lr_scheduler(optimizer=optimizer,
                                       training_config=training_config,
                                       scheduler_config=config['config']['SCHEDULERS'])

    return loss_function, optimizer, lr_scheduler

def init_epoch_dict(epoch: int, loaders, split_name: str, subsplit_name: str = 'MINIVESS') -> dict:

    # TOADD You could make this a class actually?
    results_out = {}
    for loader_key in loaders:
        epoch_dict = {
                      'scalars': {},
                      'arrays': {},
                      'metadata': {},
                      'figures': {},
                      'dataframes': {},
                     }
        results_out[loader_key] = epoch_dict

    return results_out


def collect_epoch_results(train_epoch_results, eval_epoch_results, train_results, eval_results):

    a = 1