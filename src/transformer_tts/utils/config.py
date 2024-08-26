import os

import yaml
from easydict import EasyDict as edict
from omegaconf import DictConfig, OmegaConf


def easydict_to_dict(obj):
    if not isinstance(obj, edict):
        return obj
    else:
        return {k: easydict_to_dict(v) for k, v in obj.items()}


def load_config(config_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = easydict_to_dict(config)
        config = OmegaConf.create(config)
    return config
