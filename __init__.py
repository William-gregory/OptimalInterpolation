import os

import json
import re


def get_path(*sub_dir):
    return os.path.join(os.path.dirname(__file__), *sub_dir)


def get_parent_path():
    return os.path.dirname(get_path())


def get_data_path(*sub_dir):
    return get_path('data', *sub_dir)


def get_images_path(*sub_dir):
    return get_path('images', *sub_dir)


def get_configs_path(*sub_dir):
    return get_path('configs', *sub_dir)


def read_key_from_config(config_name, key,  example=None):
    """return a key value from a config (JSON) file in configs directory"""

    # check if config_name ends with json
    if not re.search(config_name, ".json$", re.IGNORECASE):
        # add suffix if it does not
        config_name = f"{config_name}.json"
    conf_path = get_configs_path(config_name)

    if not os.path.exists(conf_path):
        print(f"config file:\n{conf_path}\ndoes not exist!")
        if example is not None:
            ex_path = get_configs_path("example_config.json")
            with open(ex_path, "r") as f:
                ex = json.load(f)
            if key in ex:
                print("expect a config file to be a JSON file with this structure")
                print({key: ex[key]})
    else:

        with open(conf_path, "r") as f:
            conf = json.load(f)

        if key in conf:
            return conf[key]
        else:
            print(f"config file:\n{conf_path}\nexists")
            print(f"however does not contain key: {key}\n"
                  f"please check that file and add the appropriate key,value pair")




