from .data_loader import *

def get_loader(arch):
    if 'patch' in arch: 
        return patch_loader
    elif 'section' in arch:
        return section_loader
    else:
        NotImplementedError()