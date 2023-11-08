from loguru import logger


def parse_tag_from_exception(e):
    # e.g.e = "Item 'minivess-segmentor:7aymxtt5skn7qs3t' already exists in the store <osfs '/home/bentoml_log/bentos'>"
    try:
        fields = e.split(' ')
        tag = fields[1]
    except Exception as e:
        logger.debug('Failed to parse tag from BentoML exception, e = {}')
    return tag