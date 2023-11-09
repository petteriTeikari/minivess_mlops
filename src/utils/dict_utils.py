from loguru import logger


def cfg_key(dct, *keys, mode: str = "RaiseError"):
    # https://stackoverflow.com/a/25833661/6412152
    for key in keys:
        if isinstance(key, tuple):
            # quick fix when safe assigning keys
            key = key[0]
        try:
            dct = dct[key]
        except KeyError:
            if mode == "NoneReturn":
                return None
            elif mode == "RaiseError":
                logger.error(
                    'Problem getting the key "{}" from the config\n{}'.format(keys, dct)
                )
                raise KeyError(
                    'Problem getting the key "{}" from the config\n{}'.format(keys, dct)
                )
            else:
                raise NotImplementedError("Unknown method = {}".format(mode))

    return dct


def safe_assign_to_dict(
    dct: dict, dict_add: dict, keys_to_check: tuple, key_to_add: str
):
    if len(keys_to_check) == 1:
        dct[keys_to_check[0]][key_to_add] = dict_add
    else:
        keys_to_check = keys_to_check[:-1]
        dct = safe_assign_to_dict(
            dct[keys_to_check[0]], dict_add, keys_to_check, key_to_add
        )

    return dct


def put_to_dict(dct, dict_add, *keys, create_parent_keys: bool = False):
    if len(keys) == 0:
        logger.error(
            "You need to provide keys when trying to add:\n{}\nto\n{}".format(
                dict_add, dct
            )
        )
        raise IOError(
            "You need to provide keys when trying to add:\n{}\nto\n{}".format(
                dict_add, dct
            )
        )

    if len(keys) == 1:
        dct[keys[0]] = dict_add

    elif len(keys) == 2:
        keys_to_check = keys[:-1]
        key_to_add = keys[-1]
        parent_dict_exists = cfg_key(dct, keys_to_check, mode="NoneReturn") is not None
        if parent_dict_exists:
            # e.g. you have dct[keys_to_check] already in dict so you can do dct[keys_to_check][key_to_add] = dict_add
            dct = safe_assign_to_dict(dct, dict_add, keys_to_check, key_to_add)
        else:
            if create_parent_keys:
                # for horrible results dictionary iteration, you could directly generate the missing
                # parent keys without having to create them in a for loop # TODO!
                raise NotImplementedError("create parent keys not implemented yet!")
            else:
                logger.error(
                    'You tried to add a key "{}" to the dictionary, '
                    'but the parent key "{}" does not exist!\n'.format(
                        key_to_add, keys_to_check
                    )
                )
                raise KeyError(
                    'You tried to add a key "{}" to the dictionary, '
                    'but the parent key "{}" does not exist!\n'.format(
                        key_to_add, keys_to_check
                    )
                )

    else:
        raise NotImplementedError("Generalize this to n-level nested dicts")

    return dct
