# MiniVess MLOps

[![Docker (Env)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/build-env_image.yml/badge.svg)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/build-env_image.yml)
[![Docker (Train)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/build_train_image.yml/badge.svg)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/build_train_image.yml)
<br>[![Test (Dataload)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/test_dataload.yml/badge.svg)](https://github.com/petteriTeikari/minivess_mlops/actions/workflows/test_dataload.yml)


(in-progress) MLOPS boilerplate code for more end-to-end reproducible pipeline for the dataset published in the article:

Charissa Poon, Petteri Teikari _et al._ (2023): 
"A dataset of rodent cerebrovasculature from in vivo multiphoton fluorescence microscopy imaging", 
Scientific Data 10, 141 doi: [10.1038/s41597-023-02048-8](https://doi.org/10.1038/s41597-023-02048-8) see also: [Dagshub](https://dagshub.com/petteriTeikari/minivess_mlops)

## Prerequisites

Tested on Ubuntu 22.04 

### Code

* Python 3.8(.17) with [Poetry](https://python-poetry.org/) environment management (instead of `venv` or `conda`)
  * [PyCharm](https://www.jetbrains.com/help/pycharm/poetry.html#poetry-env) for example knows how to create the environment from `poetry.lock` and `pyproject.toml` 
* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) (e.g. 12.2)

## Use

_This repo is **under development** still, 
and usage instructions are provided when the code is more ready for "beta release" even_

### Data

You need 2 folders set up: 1) for the DVC cache (with the input data), 2) for the output artifacts (models, figures, MLflow/WANDB temp files, etc.)

Easier to have both local development and Docker work is to symlink your local folders to the following '/mnt':

```
sudo ln -s $HOME/minivess-dvc-cache /mnt/minivess-dvc-cache
sudo ln -s $HOME/minivess-artifacts /mnt/minivess-artifacts
sudo ln -s $HOME/minivess-artifacts_local /mnt/minivess-artifacts_local
```

With the Docker, you would like the mapping to be like this (and DVC needs authentication 
to `s3://minivessdataset` where the "human-readable" Minivess dataset is, and `DVC` must 
be able to `pull`).

The mapping could be done now as in `entrypoint.sh` 
(explore the [best way to mount on startup](https://www.google.com/search?q=mount+s3+ubuntu+at+startup&oq=mount+s3+ubuntu+at+startup&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigATIKCAIQIRgWGB0YHjIKCAMQIRgWGB0YHtIBCDc2NTVqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8) 
for local development? Or try to make as frictionless the approach so that both works locally and on cloud?):

For the artifacts:

```
sudo mkdir /mnt/minivess-dvc-cache
sudo mkdir /mnt/minivess-artifacts
sudo chown $USER:$USER /mnt/minivess-artifacts /mnt/minivess-dvc-cache
mount-s3 --region eu-north-1  --allow-delete --allow-other minivess-artifacts /mnt/minivess-artifacts
```

Do you actually want to have the shared cache to be mounted 
`mount-s3 --region eu-north-1  --allow-delete --allow-other minivess-dvc-cache /mnt/minivess-dvc-cache`? 
Seems slow actually:

```
sudo ln -s ~/minivess-dvc-cache /mnt/minivess-dvc-cache
```

And if you need to manually unmount:

```
sudo umount /mnt/minivess-dvc-cache
sudo umount /mnt/minivess-artifacts
```

### Training a segmentor

Train with the defaults:

```
python run_training.py <params>
```

#### Changing the defaults

The configuration file that you wish to edit is the 
`src/configs/task_config.yaml` that was given as an example. 
This file contains only the parameters that you want to change, 
as compared to the "base config" in `src/configs/base/base_config.yaml`.
A logic for example used in the [`VISSL` library](https://colab.research.google.com/github/facebookresearch/vissl/blob/stable/tutorials/Understanding_VISSL_Training_and_YAML_Config.ipynb)

### Inference / serving

_TODO!_

## Wiki

See some background for decisions, and tutorials: https://github.com/petteriTeikari/minivess_mlops/wiki

## TODO!

See cards on [Github Projects](https://github.com/users/petteriTeikari/projects/2)
