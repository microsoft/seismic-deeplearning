  
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def normalize(array):
    """
    Normalizes a segmentation mask array to be in [0,1] range
    for use with PIL.Image
    """
    min = array.min()
    return (array - min) / (array.max() - min)

def mask_to_disk(mask, fname, cmap_name="Paired"):
    """
    write segmentation mask to disk using a particular colormap
    """
    cmap = plt.get_cmap(cmap_name)
    Image.fromarray(cmap(normalize(mask), bytes=True)).save(fname)

def image_to_disk(mask, fname, cmap_name="seismic"):
    """
    write segmentation image to disk using a particular colormap
    """
    cmap = plt.get_cmap(cmap_name)
    Image.fromarray(cmap(normalize(mask), bytes=True)).save(fname)

def decode_segmap(label_mask, colormap_name="Paired"):
    """
    Decode segmentation class labels into a colour image
        Args:
            label_mask (np.ndarray): an (N,H,W) array of integer values denoting
                the class label at each spatial location.
        Returns:
            (np.ndarray): the resulting decoded color image (NCHW).
    """
    out = np.zeros((label_mask.shape[0], 3, label_mask.shape[1], label_mask.shape[2]))
    cmap = plt.get_cmap(colormap_name)
    # loop over the batch
    for i in range(label_mask.shape[0]):
        im = Image.fromarray(cmap(normalize(label_mask[i, :, :]), bytes=True)).convert("RGB")
        out[i, :, :, :] = np.array(im).swapaxes(0, 2).swapaxes(1, 2)

    return out

def load_log_configuration(log_config_file):
    """
    Loads logging configuration from the given configuration file.
    """
    if not os.path.exists(log_config_file) or not os.path.isfile(log_config_file):
        msg = "%s configuration file does not exist!", log_config_file
        logging.getLogger(__name__).error(msg)
        raise ValueError(msg)
    try:
        logging.config.fileConfig(log_config_file, disable_existing_loggers=False)
        logging.getLogger(__name__).info("%s configuration file was loaded.", log_config_file)
    except Exception as e:
        logging.getLogger(__name__).error("Failed to load configuration from %s!", log_config_file)
        logging.getLogger(__name__).debug(str(e), exc_info=True)
        raise e


def generate_path(base_path, *directories):
    path = os.path.join(base_path, *directories)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
