import sys
from tqdm import tqdm
from loguru import logger

from src.run_training import parse_args_to_dict
from src.training.experiment import define_experiment_data
from src.utils.config_utils import import_config

# Helper subfunctions
def get_dataloaders(experim_dataloaders: dict):
    # Get the "validation" and "train" dataloaders from the dictionary
    fold_name = 'fold0'
    split_names = list(experim_dataloaders[fold_name].keys())
    fold_key = experim_dataloaders.get(fold_name)
    if fold_key is not None:
        try:
            train = experim_dataloaders[fold_name]['TRAIN']
            val = experim_dataloaders[fold_name]['VAL']['MINIVESS']
        except Exception as e:
            raise IOError('Could not get the dataloaders from the dictionary, error = {}'.format(e))
    else:
        raise IOError('Fold name = "{}" not found in the dataloaders dictionary'.format(fold_name))
    return train, val


# Input arguments for the training
input_args = ['-c', 'tutorials/train_demo']

# Fake these as coming from the command line to match the main code (run_training.py)
for sysargv in input_args:
    sys.argv.append(sysargv)
args = parse_args_to_dict()

# Create the config with Hydra from the .yaml file(s)
cfg = import_config(args=args, task_cfg_name=args['task_config_file'])

# Import the dataloaders (now the data augmentations are here as well as data transformations)
_, _, experim_dataloaders, cfg['run'] = define_experiment_data(cfg=cfg)

# Get the "validation" and "train" dataloaders from the dictionary
train, val = get_dataloaders(experim_dataloaders)

# Now you are ready to train your new model that you just wanna quickly test without
# wanting to have a battle with the config .YAML files
# Add maybe some fastai demo with MLflow autologging:
# https://github.com/mlflow/mlflow/blob/master/examples/fastai/train.py

# Iterate the dataloaders for demo
no_of_epochs = 3
logger.info('Training for {} epochs'.format(no_of_epochs))
for epoch in tqdm(range(no_of_epochs), desc='Epochs'):

    # Train
    for i, batch in enumerate(tqdm(train, desc='Training Batches')):
        images, mask = batch['image'], batch['label']

    # Validation
    for j, batch in enumerate(tqdm(val, desc='Validation Batches')):
        images, mask = batch['image'], batch['label']

logger.info('Training done!')
