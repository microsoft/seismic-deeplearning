# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

# code modified from https://github.com/waldeland/CNN-for-ASI

import torch
from torch import nn

# TODO; set chanels from yaml config file
# issue: https://github.com/microsoft/seismic-deeplearning/issues/277
class TextureNet(nn.Module):
    def __init__(self, n_classes=2):
        super(TextureNet, self).__init__()

        # Network definition
        # Parameters  #in_channels, #out_channels, filter_size, stride (downsampling factor)
        self.net = nn.Sequential(
            nn.Conv3d(1, 50, 5, 4, padding=2),
            nn.BatchNorm3d(50),
            # nn.Dropout3d() #Droput can be added like this ...
            nn.ReLU(),
            nn.Conv3d(50, 50, 3, 2, padding=1, bias=False),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            nn.Conv3d(50, 50, 3, 2, padding=1, bias=False),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            nn.Conv3d(50, 50, 3, 2, padding=1, bias=False),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            nn.Conv3d(50, 50, 3, 3, padding=1, bias=False),
            nn.BatchNorm3d(50),
            nn.ReLU(),
            nn.Conv3d(
                50, n_classes, 1, 1
            ),  # This is the equivalent of a fully connected layer since input has width/height/depth = 1
            nn.ReLU(),
        )
        # The filter weights are by default initialized by random

    def forward(self, x):
        """
        Is called to compute network output

        Args:
            x: network input - torch tensor

        Returns:
            output from the neural network

        """
        return self.net(x)

    def classify(self, x):
        """
        Classification wrapper

        Args:
            x: input tensor for classification

        Returns:
            classification result

        """
        x = self.net(x)
        _, class_no = torch.max(x, 1, keepdim=True)
        return class_no

    # Functions to get output from intermediate feature layers
    def f1(self, x):
        """
        Wrapper to obtain a particular network layer

        Args:
            x: input tensor for classification

        Returns:
            requested layer

        """
        return self.getFeatures(x, 0)

    def f2(self, x):
        """
        Wrapper to obtain a particular network layer

        Args:
            x: input tensor for classification

        Returns:
            requested layer

        """
        return self.getFeatures(x, 1)

    def f3(self, x):
        """
        Wrapper to obtain a particular network layer

        Args:
            x: input tensor for classification

        Returns:
            requested layer

        """
        return self.getFeatures(x, 2)

    def f4(self, x):
        """
        Wrapper to obtain a particular network layer

        Args:
            x: input tensor for classification

        Returns:
            requested layer

        """
        return self.getFeatures(x, 3)

    def f5(self, x):
        """
        Wrapper to obtain a particular network layer

        Args:
            x: input tensor for classification

        Returns:
            requested layer

        """
        return self.getFeatures(x, 4)

    def getFeatures(self, x, layer_no):
        """
        Main call method to call the wrapped layers

        Args:
            x: input tensor for classification
            layer_no: number of hidden layer we want to extract

        Returns:
            requested layer

        """
        layer_indexes = [0, 3, 6, 9, 12]

        # Make new network that has the layers up to the requested output
        tmp_net = nn.Sequential()
        layers = list(self.net.children())[0 : layer_indexes[layer_no] + 1]
        for i in range(len(layers)):
            tmp_net.add_module(str(i), layers[i])
        return tmp_net(x)


def get_seg_model(cfg, **kwargs):
    model = TextureNet(n_classes=cfg.DATASET.NUM_CLASSES)
    return model
