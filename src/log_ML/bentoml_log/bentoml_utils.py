import json
import subprocess

from loguru import logger


def parse_tag_from_exception(e):
    # e.g.e = "Item 'minivess-segmentor:7aymxtt5skn7qs3t' already exists in the store <osfs '/home/bentoml_log/bentos'>"
    try:
        fields = e.split(' ')
        tag = fields[1]
    except Exception as e:
        logger.debug('Failed to parse tag from BentoML exception, e = {}')
    return tag


def get_latest_bento_docker(docker_image: str,
                            run_id_tag: str,
                            host: str = 'https://registry.hub.docker.com/v2/repositories'):

    def get_non_latest_tag(json_object: dict):
        images_out, tags = [], []
        for image in json_object['results']:
            if image['name'] != 'latest':
                logger.debug('Found tag (run_id) = {}'.format(image['name']))
                images_out.append(image)
                tags.append(image['name'])
        logger.info('Found a total of {} non-"latest" tags'.format(len(images_out)))
        return images_out, tags

    # import docker
    # try:
    #     # Installation error occurred, docker seemes to be installed via "poetry add",
    #     # but still cannot use it?
    #     client = docker.from_env()
    # except Exception as e:
    #     logger.debug(' docker.__path__ = {}'.format(docker.__path__))
    #     logger.error('Failed to initialize Docker (Python) client, e = {}'.format(e))
    #     # OSError: Failed to initialize Docker (Python) client, e = module 'docker' has no attribute 'from_env'
    #     raise OSError('Failed to initialize Docker (Python) client, e = {}'.format(e))

    # Use the Docker CLI API with jq (sudo apt install jq)
    # https://forums.docker.com/t/how-can-i-list-tags-for-a-repository/32577/10
    # curl 'https://registry.hub.docker.com/v2/repositories/petteriteikari/minivess-segmentor/tags/'|jq '."results"[]["name"]'

    hub_path = f'{host}/{docker_image}/tags/'
    command = f'curl "{hub_path}"|jq'
    data = subprocess.run(command, capture_output=True, shell=True, text=True)
    json_object = json.loads(data.stdout)

    images_out, tags = get_non_latest_tag(json_object)

    if run_id_tag in tags:
        docker_built_from_best_run_id = True
    else:
        last_push = images_out[0]['tag_last_pushed']
        logger.info('Bento image on Docker Hub seems to be behind the best model on MLFlow, '
                    'Last push = {}'.format(last_push))
        docker_built_from_best_run_id = False

    return docker_built_from_best_run_id


def get_bento_CLI_build_command(containarize: bool = False,
                                ):

    # TODO! add the no-cache option to the build command
    if containarize:
        containarize_string = '--containerize'
    else:
        containarize_string = ''
    cmd = f'bentoml build -f bentofile.yaml {containarize_string} .'
    logger.info('Building Bento locally, cmd = "{}"'.format(cmd))

    return cmd


def get_bento_CLI_containerize_command(bento_tag: str,
                                       no_cache: bool = False):
    # e.g. "bentoml containerize minivess-segmentor:a6bgb6d3nob4nj6o --opt no-cache"
    if no_cache:
        cache_string = ' --opt no-cache'
    else:
        logger.warning('You are containerizing the Bento with cache,\nyou might easily get some cached garbage and wonder why your Docker does not behave as you think it should based on your code.\nUse with caution!')
        cache_string = ''
    cmd = f'bentoml containerize {bento_tag}{cache_string}'
    logger.info('Containarize Bento locally, cmd = "{}"'.format(cmd))

    return cmd