
## Python 3.8(.17)

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.8
sudo apt install python3.8-distutils
```

## Poetry remote ssh env

### Pycharm

Not working
https://youtrack.jetbrains.com/issue/PY-52688


## Poetry for VSCode

Run?

```
poetry config virtualenvs.in-project true
```

### SSH

https://code.visualstudio.com/docs/remote/remote-overview
https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack


## Licublas issue with Poetry

Fixed for Poetry 
https://stackoverflow.com/questions/76327419/valueerror-libcublas-so-0-9-not-found-in-the-system-path

```
ValueError: libcublas.so.*[0-9] not found in the system path ['/home/petteri/PycharmProjects/minivess_mlops/src', '/snap/pycharm-professional/346/plugins/python/helpers/pydev', '/snap/pycharm-professional/346/plugins/python/helpers/third_party/thriftpy', '/snap/pycharm-professional/346/plugins/python/helpers/pydev', '/home/petteri/PycharmProjects/minivess_mlops', '/snap/pycharm-professional/346/plugins/python/helpers/pycharm_display', '/home/petteri/.cache/JetBrains/PyCharm2023.2/cythonExtensions', '/home/petteri/PycharmProjects/minivess_mlops/src', '/usr/lib/python38.zip', '/usr/lib/python3.8', '/usr/lib/python3.8/lib-dynload', '/home/petteri/.cache/pypoetry/virtualenvs/mlops-nuJNTY7i-py3.8/lib/python3.8/site-packages', '/snap/pycharm-professional/346/plugins/python/helpers/pycharm_matplotlib_backend']
```

## WANDB

- Signup to to WANDB
 
- Install WANDB: `pip install wandb`

- Login to WANBD: `wandb login`

```bash
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit: 
wandb: Appending key for api.wandb.ai to your netrc file: /home/petteri/.netrc
```

## MLflow

```
mlflow ui --backend-store-uri ./mlruns
```