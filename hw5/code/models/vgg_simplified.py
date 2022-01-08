import torch
import torch.nn as nn
import math


class ConvReLUMaxPool2d(nn.Module):
    def __init__(
        self, 
        in_c, 
        out_c, 
        conv_kernel_size=3, 
        pooling_kernel_size=2
    ):
        super(ConvReLUMaxPool2d, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c, 
                out_channels=out_c, 
                kernel_size=conv_kernel_size, 
                padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=pooling_kernel_size)
        )
    def forward(self, x):
        return self.layers(x)


class Vgg(nn.Module):
    def __init__(self, fc_layer=512, classes=10):
        super(Vgg, self).__init__()
        """ Initialize VGG simplified Module
        Args: 
            fc_layer: input feature number for the last fully MLP block
            classes: number of image classes
        """
        self.fc_layer = fc_layer
        self.classes = classes

        # todo: construct the simplified VGG network blocks
        # input shape: [bs, 3, 32, 32]
        # layers and output feature shape for each block:
        # # conv_block1 (Conv2d, ReLU, MaxPool2d) --> [bs, 64, 16, 16]
        # # conv_block2 (Conv2d, ReLU, MaxPool2d) --> [bs, 128, 8, 8]
        # # conv_block3 (Conv2d, ReLU, MaxPool2d) --> [bs, 256, 4, 4]
        # # conv_block4 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 2, 2]
        # # conv_block5 (Conv2d, ReLU, MaxPool2d) --> [bs, 512, 1, 1]
        # # classifier (Linear, ReLU, Dropout2d, Linear) --> [bs, 10] (final output)

        # hint: stack layers in each block with nn.Sequential, e.x.:
        # # self.conv_block1 = nn.Sequential(
        # #     layer1,
        # #     layer2,
        # #     layer3,
        # #     ...)

        self.layers = nn.Sequential(
            ConvReLUMaxPool2d(3, 64),
            ConvReLUMaxPool2d(64, 128),
            ConvReLUMaxPool2d(128, 256),
            ConvReLUMaxPool2d(256, 512),
            ConvReLUMaxPool2d(512, 512),

            nn.Flatten(),
            nn.Linear(512, self.fc_layer),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.fc_layer, self.classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        """
        :param x: input image batch tensor, [bs, 3, 32, 32]
        :return: score: predicted score for each class (10 classes in total), [bs, 10]
        """
        return self.layers(x)

