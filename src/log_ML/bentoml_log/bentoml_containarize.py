import bentoml



def containarize_bento(bento_model: bentoml._internal.models.model.Model,
                       docker_image: str):
    """
    Bento build:
    bentoml build -f bentofile.yaml --containerize  .

    Rename created docker image:
    docker tag minivess-segmentor:dmp57wd55kn7qs3t petteriteikari/minivess-segmentor:latest

    # delete old tag
    docker image rm minivess-segmentor:dmp57wd55kn7qs3t

    Add additional tag that is the run_id of the MLFlow model so you could automatically check if
    the model has improved and you would need to re-build the serving Docker as well:
    docker tag petteriteikari/minivess-segmentor:latest petteriteikari/minivess-segmentor:125868ff569a406399cdd8e35c3a0cda

    Push to Docker Hub (both two tags, after deleting the first tag):
    docker image push --all-tags petteriteikari/minivess-segmentor
    """

    a = 1