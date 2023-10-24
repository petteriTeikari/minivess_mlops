import os

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

cwd = os.getcwd()
repo_dir = os.path.join(cwd, '..', '..')
if not os.path.exists(repo_dir):
    raise IOError(f"repo_dir: {repo_dir} does not exist")

DEFAULTS_NAME = 'defaults'
DEFAULTS = f'{DEFAULTS_NAME}.yaml'
DEFAULT_DIR = os.path.join(repo_dir, 'configs')
DEFAULT_DIR_RELATIVE = os.path.join('..', '..', 'configs')
if not os.path.exists(DEFAULT_DIR):
    raise IOError(f"DEFAULT_DIR: {DEFAULT_DIR} does not exist")

DEFAULTS_PATH = os.path.join(DEFAULT_DIR, DEFAULTS)
if not os.path.exists(DEFAULTS_PATH):
    raise IOError(f"DEFAULTS_PATH: {DEFAULTS_PATH} does not exist")

TASK_NAME = 'test_dataload_config'
TASK_CONFIG_FILE = f'{TASK_NAME}.yaml'
TASK_DIR = os.path.join(DEFAULT_DIR, 'runmode_configs')
if not os.path.exists(TASK_DIR):
    raise IOError(f"TASK_DIR: {TASK_DIR} does not exist")

TASK_PATH = os.path.join(TASK_DIR, TASK_CONFIG_FILE)
if not os.path.exists(TASK_PATH):
    raise IOError(f"TASK_PATH: {TASK_PATH} does not exist")

TASK_EX = os.path.join(TASK_DIR, TASK_NAME) # without the .yaml


# See e.g. https://stackoverflow.com/a/77147018
#          https://hydra.cc/docs/patterns/configuring_experiments/
# https://github.com/facebookresearch/vissl/blob/94def58538d3c7037f5e093196494331eea1a2a2/tools/run_distributed_engines.py#L55

# How not needing to use the .main() syntax:
# https://stackoverflow.com/a/61169706/6412152
# https://stackoverflow.com/questions/60674012/how-to-get-a-hydra-config-without-using-hydra-main
# https://stackoverflow.com/questions/73748800/how-to-use-hydra-config-alongside-user-defined-parameters?rq=3

# @hydra.main(version_base='1.2', config_path=DEFAULT_DIR, config_name=DEFAULTS_NAME)
# def my_app(cfg: DictConfig) -> None:
#     print(OmegaConf.to_yaml(cfg))

def import_config_hydra():

    # https://stackoverflow.com/a/61169706/6412152
    # init with the dir where your defaults.yaml is
    with initialize(config_path=DEFAULT_DIR_RELATIVE,
                    job_name="test_hydra_minivess",
                    version_base='1.2'):

        # compose the config
        args_task_cfg_file = TASK_EX
        # https://stackoverflow.com/a/77147018
        # cfg = compose(config_name=DEFAULTS_NAME, overrides=[f'config={args_task_cfg_file}'])
        #cfg = compose(config_name='config', overrides=['+fast_learn=c'])
        cfg = compose(config_name='defaults', overrides=['+runmode_configs=test_dataload_config'])

    return cfg


if __name__ == "__main__":
    # as our run_training-py
    cfg = import_config_hydra()