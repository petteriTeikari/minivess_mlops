# minivess_mlops

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

Create local dir `minivess_mlops_artifacts` e.g. to `~` (Home directory) and symlink that to 
a mount point (so you have the same path for Docker and local devel):

```
sudo ln -s ~/minivess_mlops_artifacts/ /mnt/minivess_mlops_artifacts
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

At the moment using the [`OmegaConf`](https://omegaconf.readthedocs.io/) for managing configuration dictionary, see later if some [`Hydra`](https://hydra.cc/) is needed on top of it. 

### Inference / serving

_TODO!_