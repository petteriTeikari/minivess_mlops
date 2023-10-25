import os
import sys
import pytest

from loguru import logger

# quick fix to get the src. imports working
src_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(os.path.split(src_path)[0])[0]
sys.path.insert(0, project_path)

from src.utils.config_utils import config_import_script

a = 1
CFG_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'configs')
if not os.path.exists(CFG_DIR):
    raise IOError('Config directory "{}" does not exist!'.format(CFG_DIR))


def get_config_files(cfg_dir: str,
                     exclude_files: tuple = ('defaults.yaml'),
                     extension: str = 'yaml'):
    abs_config_files, relative_cfg_paths = [], []
    for root, dirs, files in os.walk(cfg_dir):
        for file in files:
            if file.endswith(extension):
                if file not in exclude_files:
                    abs_path = os.path.join(root, file)
                    rel_path = abs_path.replace(cfg_dir+os.sep, '').replace('.'+extension, '')
                    abs_config_files.append(abs_path)
                    relative_cfg_paths.append(rel_path)
    logger.info('Found a total of {} "task" config files'.format(len(abs_config_files)))

    return abs_config_files, relative_cfg_paths


def test_import_configs(cfg_dir: str = None,
                        exclude_files: tuple = ('defaults.yaml', )):

    if cfg_dir is None:
        cfg_dir = CFG_DIR
    else:
        if not os.path.exists(cfg_dir):
            raise IOError('Config directory "{}" does not exist!'.format(cfg_dir))

    # TO-OPIMIZE, check the use of "fixtures" here
    # https://docs.pytest.org/en/6.2.x/fixture.html
    try:
        abs_config_files, relative_cfg_paths = get_config_files(cfg_dir=cfg_dir, exclude_files=exclude_files)
    except Exception as e:
        raise IOError('Problem getting the config files, e = {}'.format(e))

    any_config_failed = False
    try:
        i = 0
        for defaults_cfg in exclude_files:
            for relative_cfg_path in relative_cfg_paths:
                i += 1
                logger.info('#{}: Testing cfg = {} (base = {})'.format(i, relative_cfg_path, defaults_cfg))
                try:
                    config = config_import_script(task_cfg_name=relative_cfg_path,
                                                  base_cfg_name=defaults_cfg.replace('.yaml', ''),
                                                  parent_dir_string = '.', #f'..{os.sep}..',
                                                  parent_dir_string_defaults = f'..{os.sep}..')
                except Exception as e:
                    logger.error('Fail to import config "{}", e = {}'.format(relative_cfg_path, e))
                    any_config_failed = True

    except Exception as e:
        raise IOError('Problem with the config files, e = {}'.format(e))

    assert any_config_failed == False

test_import_configs()