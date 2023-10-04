import yaml


def import_yaml_file(yaml_path: str):

    # TOADD! add scientifc notation resolver? e.g. for lr https://stackoverflow.com/a/30462009/6412152
    with open(yaml_path) as file:
        try:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)

    if 'cfg' not in locals():
        raise IOError('YAML import failed! See the the line and columns above that were not parsed correctly!\n'
                      '\t\tI assume that you added or modified some entries and did something illegal there?\n'
                      '\t\tMaybe a "=" instead of ":"?\n'
                      '\t\tMaybe wrong use of "’" as the closing quote?')

    return cfg
