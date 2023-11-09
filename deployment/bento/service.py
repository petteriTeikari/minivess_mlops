import os
import sys
import bentoml

bentoml_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.split(os.path.split(bentoml_path)[0])[0]
sys.path.insert(0, project_path)

# Hard-coded at the moment in bentoml_utils.py for testing purposes
MODEL_NAME = "minivess-segmentor"

minivess_runner = bentoml.mlflow.get(f"{MODEL_NAME}:latest").to_runner()

svc = bentoml.Service("minivess-segmentor", runners=[minivess_runner])

input_spec = bentoml.io.NumpyNdarray(
    dtype="float32",
    shape=(-1, 1, -1, -1, -1),
    enforce_shape=True,
    enforce_dtype=True,
)


@svc.api(input=input_spec, output=bentoml.io.NumpyNdarray())
def predict(input_arr):
    return minivess_runner.predict.run(input_arr)
