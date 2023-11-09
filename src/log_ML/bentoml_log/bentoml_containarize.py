import os
import subprocess
import time

from loguru import logger
import bentoml

from src.log_ML.bentoml_log.bentoml_utils import get_bento_CLI_containerize_command
from src.utils.config_utils import get_repo_dir


def log_docker_build_std_out_err(stdout: str, stderr: str):
    logger.info(stderr)
    logger.info(stdout)

    lines = stdout.split("\n")
    tag = None
    for line in lines:
        if "Successfully built" in line:
            split_fields = line.split(" ")
            tag = split_fields[-1].replace('"', "")

    if tag is not None:
        docker_name, tag = tag.split(":")
        logger.info('Built Docker "{}" with tag = "{}"'.format(docker_name, tag))

    return docker_name, tag


def modify_docker_image_tags(
    docker_name: str, tag: str, docker_image: str, run_id: str = None
):
    """
    Rename created docker image:
    docker tag minivess-segmentor:dmp57wd55kn7qs3t petteriteikari/minivess-segmentor:latest

    # delete old tag
    docker image rm minivess-segmentor:dmp57wd55kn7qs3t

    Add additional tag that is the run_id of the MLFlow model so you could automatically check if
    the model has improved and you would need to re-build the serving Docker as well:
    docker tag petteriteikari/minivess-segmentor:latest petteriteikari/minivess-segmentor:125868ff569a406399cdd8e35c3a0cda
    """

    docker_image_out = f"{docker_image}:latest"
    cmd = f"docker tag {docker_name}:{tag} {docker_image_out}"
    logger.info('Add the "latest" tag, cmd = "{}"'.format(cmd))
    out = subprocess.run(cmd, capture_output=True, shell=True, text=True)

    cmd = f"docker image rm {docker_name}:{tag}"
    logger.info('Remove the original autocreated tag, cmd = "{}"'.format(cmd))
    out = subprocess.run(cmd, capture_output=True, shell=True, text=True)

    docker_image_run_id = f"{docker_image}:{run_id}"
    cmd = f"docker tag {docker_image_out} {docker_image_run_id}"
    logger.info(
        'Adding another tag (MLFlow run_id) to the Docker image, cmd = "{}"'.format(cmd)
    )
    out = subprocess.run(cmd, capture_output=True, shell=True, text=True)


def push_bento_to_docker_hub(docker_image: str):
    cmd = f"docker image push --all-tags {docker_image}"
    logger.info('Push to Docker Hub, cmd = "{}"'.format(cmd))
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, shell=True, text=True)
    logger.info(out.stdout)
    logger.info(out.stderr)
    logger.info(
        "Bento Docker upload to Docker Hub took {:.0f} seconds".format(time.time() - t0)
    )


def containarize_bento(
    bento_tag: str, docker_image: str, run_id: str, no_cache: bool = True
):
    """

    Push to Docker Hub (both two tags, after deleting the first tag):
    docker image push --all-tags petteriteikari/minivess-segmentor
    """

    # set working directory to the root of the repo
    cwd = os.getcwd()
    repo_dir = get_repo_dir()
    os.chdir(repo_dir)

    # get build command
    cmd = get_bento_CLI_containerize_command(bento_tag=bento_tag, no_cache=no_cache)
    logger.info('Start building Bento container, cmd = "{}"'.format(cmd))
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, shell=True, text=True)
    docker_name, tag = log_docker_build_std_out_err(
        stdout=out.stdout, stderr=out.stderr
    )
    logger.info("Bento container build took {:.0f} seconds".format(time.time() - t0))

    # Add "run_id" tag so you can see from Docker Hub clearly what MLFlow run_id produced the Docker image
    modify_docker_image_tags(
        docker_name=docker_name, tag=tag, docker_image=docker_image, run_id=run_id
    )

    # Push to Docker Hub
    push_bento_to_docker_hub(docker_image=docker_image)

    # Use original working directory
    os.chdir(cwd)
