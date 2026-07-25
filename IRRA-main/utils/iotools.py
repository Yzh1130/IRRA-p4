# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from PIL import Image, ImageFile
import errno
import json
import pickle as pkl
import os
import os.path as osp
import yaml
from easydict import EasyDict as edict
from utils.config import ConfigLoadError, load_config_mapping

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    # Retry loop helps when storage IO is temporarily unstable.
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def mkdir_if_missing(directory):
    # Safe mkdir that tolerates concurrent creation.
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    # Return whether file exists and warn when missing.
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    # Read JSON file into Python object.
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    # Write JSON to disk with pretty indent.
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def get_text_embedding(path, length):
    # Placeholder for loading cached text embedding frequency data.
    with open(path, 'rb') as f:
        word_frequency = pkl.load(f)


def save_train_configs(path, args):
    # Persist argparse namespace as YAML for reproducible inference.
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/configs.yaml', 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def load_train_configs(path):
    # Load YAML config and expose dict fields as attributes.
    return edict(load_config_mapping(path))
