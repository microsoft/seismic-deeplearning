# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np

def _chw_to_hwc(image_array_numpy):
    return np.moveaxis(image_array_numpy, 0, -1)


