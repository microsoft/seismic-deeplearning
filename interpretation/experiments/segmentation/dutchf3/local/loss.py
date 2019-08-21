import torch
import torch.nn.functional as F
from toolz import curry

@curry
def cross_entropy(prediction, target, weight=None, ignore_index=255):
    '''
    Use 255 to fill empty values when padding or doing any augmentation operations
    like rotation. 
    '''
    loss = F.cross_entropy(prediction, target, weight, reduction='mean',  ignore_index=255)
    return loss
