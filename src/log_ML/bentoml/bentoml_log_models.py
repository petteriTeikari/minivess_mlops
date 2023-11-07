import os
import subprocess

from src.inference.bentoml_utils import import_mlflow_model_to_bentoml
from loguru import logger

def bentoml_save_mlflow_model_to_model_store(run,
                                             ensemble_name: str = 'dice-MINIVESS',
                                             model_uri: str = 'models:/minivess-test2/1',
                                             bento_model_name: str = 'minivess-segmentor'):

    # see, "bentoml list" for the saved model (as you would see your Docker images)
    bento_model, pyfunc_model = import_mlflow_model_to_bentoml(run=run,
                                                               model_uri=model_uri,
                                                               bento_model_name=bento_model_name)

    logger.info('BentoML Model | tag = {}'.format(bento_model.tag))
    logger.info('BentoML Model | path = {}'.format(bento_model.path))
    model_yaml_path = os.path.join(bento_model.path, 'model.yaml')
    if os.path.exists(model_yaml_path):
        logger.info('BentoML Model | model.yaml path = {}'.format(model_yaml_path))
    logger.info('See output of "bentoml models list"')
    p = subprocess.run(["bentoml", "models", "list"], capture_output=True, text=True)
    logger.info(p.stdout)
    logger.info(f'You can use the tag "latest" with the model: {bento_model_name}:latest')
    # add the actual docker building, s3 export option?

    return bento_model, pyfunc_model