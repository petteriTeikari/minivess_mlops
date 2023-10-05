# minivess_mlops

(in-progress) MLOPS boilerplate code for more end-to-end reproducible pipeline for the dataset published in the article:

Charissa Poon, Petteri Teikari _et al._ (2023): 
"A dataset of rodent cerebrovasculature from in vivo multiphoton fluorescence microscopy imaging", 
Scientific Data 10, 141 doi: [10.1038/s41597-023-02048-8](https://doi.org/10.1038/s41597-023-02048-8)

## Prerequisites

Tested on Ubuntu 22.04 

### Code

* Python 3.8(.17) with [Poetry](https://python-poetry.org/) environment management (instead of `venv` or `conda`)
  * [PyCharm](https://www.jetbrains.com/help/pycharm/poetry.html#poetry-env) for example knows how to create the environment from `poetry.lock` and `pyproject.toml` 
* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) (e.g. 12.2)
* _TODO!_ Create Dockerfile

## Use

### Data

Create local dir `minivess_mlops_artifacts` e.g. to `~` (Home directory) and symlink that to 
a mount point (so you have the same path for Docker and local devel):

```
sudo ln -s ~/minivess_mlops_artifacts/ /mnt/minivess_mlops_artifacts
```

* Script should automatically download the data (This seems to require authentication with the EBrains API?)
* (backup option) Download data manually from EBRAINS: https://doi.org/10.25493/HPBE-YHK, and copy `d-bf268b89-1420-476b-b428-b85a913eb523.zip` to `/mnt/minivess_mlops_artifacts/data`


### Training a segmentor

Train with the defaults:

```
python run_training.py -c base_config.yaml
```

#### Changing the defaults

The configuration file that you wish to edit is the 
`src/configs/task_config.yaml` that was given as an example. 
This file contains only the parameters that you want to change, 
as compared to the "base config" in `src/configs/base/base_config.yaml`.
A logic for example used in the [`VISSL` library](https://colab.research.google.com/github/facebookresearch/vissl/blob/stable/tutorials/Understanding_VISSL_Training_and_YAML_Config.ipynb)

At the moment using the [`OmegaConf`](https://omegaconf.readthedocs.io/) for managing configuration dictionary, see later if some [`Hydra`](https://hydra.cc/) is needed on top of it. 

### Inference / serving

## Features/tasks to do

* ML Tests / CI/CD (e.g. [https://github.com/recommenders-team/recommenders/pull/1989](https://github.com/recommenders-team/recommenders/pull/1989))
* Experiment tracking (WANDB and MLFlow)
* Data versioning (DVC)
  * Where to actually get the data as EBrains is not the most user-friendly. Add some Airflow for simulating if you had one images coming from experiments all the time?
* Integrate stuff with Daqshub?
* Demo ML Experiment with some off-the-shelf MONAI segmentor against arXiv/github 3rd party methods so that you get an example how to add stuff from papers to this pipeline
  * e.g. [Butoi et al. 2023: "UniverSeg"](https://universeg.csail.mit.edu/) and some medical SAM variant like [Lei et al. (2023): "MedLSAM: Localize and Segment Anything Model for 3D Medical Images"](https://github.com/openmedlab/MedLSAM)
* Training script
  * Add additional nesting allowing "diverse ensembles" so that each submodel of the ensemble could be of a different architecture
  * Add some sort of hierarchical model training allowing the training of "foundational model" with its finetuning on one go (most likely just freeze the foundation model and finetune it yourself)
* Need to add an additional dataset to demonstrate how to train/evaluate on multiple separate subsets of the whole training corpus