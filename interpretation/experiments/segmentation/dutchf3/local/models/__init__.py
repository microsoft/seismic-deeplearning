import torchvision.models as models
from .patch_deconvnet import *
from .section_deconvnet import *

def get_model(name, pretrained, n_classes):
    model = _get_model_instance(name)

    if name in ['section_deconvnet','patch_deconvnet']:
        model = model(n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=pretrained)
        model.init_vgg16_params(vgg16)
    else:
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'section_deconvnet': section_deconvnet,
            'patch_deconvnet':patch_deconvnet,
            'section_deconvnet_skip': section_deconvnet_skip,
            'patch_deconvnet_skip':patch_deconvnet_skip,
        }[name]
    except:
        print(f'Model {name} not available')
