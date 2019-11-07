# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
from deepseismic_interpretation.dutchf3.data import decode_segmap
from os import path
from PIL import Image
from toolz import pipe


def _chw_to_hwc(image_array_numpy):
    return np.moveaxis(image_array_numpy, 0, -1)


def save_images(pred_dict, output_dir, num_classes, colours, extra_identifier=""):
    for id in pred_dict:
        save_image(
            pred_dict[id].unsqueeze(0).cpu().numpy(),
            output_dir,
            num_classes,
            colours,
            extra_identifier=extra_identifier,
        )


def save_image(
    image_numpy_array, output_dir, num_classes, colours, extra_identifier=""
):
    """Save segmentation map as image
    
    Args:
        image_numpy_array (numpy.Array): numpy array that represents an image
        output_dir ([type]): 
        num_classes ([type]): [description]
        colours ([type]): [description]
        extra_identifier (str, optional): [description]. Defaults to "".
    """
    im_array = decode_segmap(
        image_numpy_array, n_classes=num_classes, label_colours=colours,
    )
    im = pipe(
        (im_array * 255).astype(np.uint8).squeeze(), _chw_to_hwc, Image.fromarray,
    )
    filename = path.join(output_dir, f"{id}_{extra_identifier}.png")
    im.save(filename)
