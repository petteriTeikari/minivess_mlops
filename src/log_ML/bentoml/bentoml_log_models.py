import os

from src.inference.bentoml_utils import import_mlflow_model_to_bentoml
from loguru import logger

def bentoml_save_mlflow_model_to_model_store(run,
                                             model_uri: str = 'models:/minivess-test2/1'):

    # see, "bentoml list" for the saved model (as you would see your Docker images)
    bento_model, pyfunc_model = import_mlflow_model_to_bentoml(run=run,
                                                               model_uri=model_uri)

    logger.info('BentoML Model | tag = {}'.format(bento_model.tag))
    logger.info('BentoML Model | path = {}'.format(bento_model.path))
    model_yaml_path = os.path.join(bento_model.path, 'model.yaml')
    if not os.path.exists(model_yaml_path):
        logger.info('BentoML Model | model.yaml path = {}'.format(bento_model.path))
    logger.info(' see output of "bentoml list"')
    # add the actual docker building, s3 export option?

    return bento_model, pyfunc_model