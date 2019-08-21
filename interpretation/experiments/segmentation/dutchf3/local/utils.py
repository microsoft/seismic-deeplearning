import numpy as np
import torch 
import torchvision.utils as vutils

def np_to_tb(array):
    # if 2D :
    if array.ndim == 2:
        # HW => CHW
        array = np.expand_dims(array,axis=0)
        # CHW => NCHW
        array = np.expand_dims(array,axis=0)
    elif array.ndim == 3:
        # HWC => CHW
        array = array.transpose(2, 0, 1)
        # CHW => NCHW
        array = np.expand_dims(array,axis=0)
    
    array = torch.from_numpy(array)
    array = vutils.make_grid(array, normalize=True, scale_each=True)
    return array
