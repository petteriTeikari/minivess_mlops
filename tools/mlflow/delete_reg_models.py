from pprint import pprint
from mlflow import MlflowClient

# Add authentication to the client

client = MlflowClient()
rmodels = client.search_registered_models()

for rm in rmodels:
    pprint(dict(rm), indent=4)
    # https://mlflow.org/docs/latest/model-registry.html#deleting-mlflow-models
    print('Deleting model "{}"'.format(rm.name))
    try:
        client.delete_registered_model(name=rm.name)
    except Exception as e:
        print("Failed to delete model! e = {}".format(e))
