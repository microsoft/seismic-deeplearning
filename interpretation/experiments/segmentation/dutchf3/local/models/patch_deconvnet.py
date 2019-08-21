import torch.nn as nn

class patch_deconvnet(nn.Module):

    def __init__(self, n_classes=4, learned_billinear=False):
        super(patch_deconvnet, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv_block1 = nn.Sequential(

            # conv1_1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv1_2
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool1
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_1

        # 48*48

        self.conv_block2 = nn.Sequential(

            # conv2_1
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv2_2
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool2
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_2

        # 24*24

        self.conv_block3 = nn.Sequential(

            # conv3_1
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv3_2
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv3_3
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool3
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_3

        # 12*12

        self.conv_block4 = nn.Sequential(

            # conv4_1
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv4_2
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv4_3
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool4
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_4

        # 6*6

        self.conv_block5 = nn.Sequential(

            # conv5_1
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv5_2
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv5_3
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool5
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_5

        # 3*3

        self.conv_block6 = nn.Sequential(

            # fc6
            nn.Conv2d(512, 4096, 3),
            # set the filter size and nor padding to make output into 1*1
            nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        # 1*1

        self.conv_block7 = nn.Sequential(

            # fc7
            nn.Conv2d(4096, 4096, 1),
            # set the filter size to make output into 1*1
            nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.deconv_block8 = nn.Sequential(

            # fc6-deconv
            nn.ConvTranspose2d(4096, 512, 3, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        # 3*3

        self.unpool_block9 = nn.Sequential(

            # unpool5
            nn.MaxUnpool2d(2, stride=2), )
        # usage unpool(output, indices)

        # 6*6

        self.deconv_block10 = nn.Sequential(

            # deconv5_1
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv5_2
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv5_3
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block11 = nn.Sequential(

            # unpool4
            nn.MaxUnpool2d(2, stride=2), )

        # 12*12

        self.deconv_block12 = nn.Sequential(

            # deconv4_1
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv4_2
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv4_3
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block13 = nn.Sequential(

            # unpool3
            nn.MaxUnpool2d(2, stride=2), )

        # 24*24

        self.deconv_block14 = nn.Sequential(

            # deconv3_1
            nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv3_2
            nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv3_3
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block15 = nn.Sequential(

            # unpool2
            nn.MaxUnpool2d(2, stride=2), )

        # 48*48

        self.deconv_block16 = nn.Sequential(

            # deconv2_1
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv2_2
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block17 = nn.Sequential(

            # unpool1
            nn.MaxUnpool2d(2, stride=2), )

        # 96*96

        self.deconv_block18 = nn.Sequential(

            # deconv1_1
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv1_2
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.seg_score19 = nn.Sequential(

            # seg-score
            nn.Conv2d(64, self.n_classes, 1), )

        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        size0 = x.size()
        conv1, indices1 = self.conv_block1(x)
        size1 = conv1.size()
        conv2, indices2 = self.conv_block2(conv1)
        size2 = conv2.size()
        conv3, indices3 = self.conv_block3(conv2)
        size3 = conv3.size()
        conv4, indices4 = self.conv_block4(conv3)
        size4 = conv4.size()
        conv5, indices5 = self.conv_block5(conv4)

        conv6 = self.conv_block6(conv5)
        conv7 = self.conv_block7(conv6)
        conv8 = self.deconv_block8(conv7)
        conv9 = self.unpool(conv8, indices5, output_size=size4)
        conv10 = self.deconv_block10(conv9)
        conv11 = self.unpool(conv10, indices4, output_size=size3)
        conv12 = self.deconv_block12(conv11)
        conv13 = self.unpool(conv12, indices3, output_size=size2)
        conv14 = self.deconv_block14(conv13)
        conv15 = self.unpool(conv14, indices2, output_size=size1)
        conv16 = self.deconv_block16(conv15)
        conv17 = self.unpool(conv16, indices1, output_size=size0)
        conv18 = self.deconv_block18(conv17)
        out = self.seg_score19(conv18)

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        i_layer = 0;
        # copy convolutional filters from vgg16
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    if i_layer == 0:
                        l2.weight.data = ((l1.weight.data[:, 0, :, :] + l1.weight.data[:, 1, :, :] + l1.weight.data[:,
                                                                                                     2, :,
                                                                                                     :]) / 3.0).view(
                            l2.weight.size())
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1
                    else:
                        assert l1.weight.size() == l2.weight.size()
                        assert l1.bias.size() == l2.bias.size()
                        l2.weight.data = l1.weight.data
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1


class patch_deconvnet_skip(nn.Module):

    def __init__(self, n_classes=4, learned_billinear=False):
        super(patch_deconvnet_skip, self).__init__()
        self.learned_billinear = learned_billinear
        self.n_classes = n_classes
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv_block1 = nn.Sequential(

            # conv1_1
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv1_2
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool1
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_1

        # 48*48

        self.conv_block2 = nn.Sequential(

            # conv2_1
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv2_2
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool2
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_2

        # 24*24

        self.conv_block3 = nn.Sequential(

            # conv3_1
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv3_2
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv3_3
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool3
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_3

        # 12*12

        self.conv_block4 = nn.Sequential(

            # conv4_1
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv4_2
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv4_3
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool4
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_4

        # 6*6

        self.conv_block5 = nn.Sequential(

            # conv5_1
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv5_2
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # conv5_3
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # pool5
            nn.MaxPool2d(2, stride=2, return_indices=True, ceil_mode=True), )
        # it returns outputs and pool_indices_5

        # 3*3

        self.conv_block6 = nn.Sequential(

            # fc6
            nn.Conv2d(512, 4096, 3),
            # set the filter size and nor padding to make output into 1*1
            nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        # 1*1

        self.conv_block7 = nn.Sequential(

            # fc7
            nn.Conv2d(4096, 4096, 1),
            # set the filter size to make output into 1*1
            nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.deconv_block8 = nn.Sequential(

            # fc6-deconv
            nn.ConvTranspose2d(4096, 512, 3, stride=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        # 3*3

        self.unpool_block9 = nn.Sequential(

            # unpool5
            nn.MaxUnpool2d(2, stride=2), )
        # usage unpool(output, indices)

        # 6*6

        self.deconv_block10 = nn.Sequential(

            # deconv5_1
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv5_2
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv5_3
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block11 = nn.Sequential(

            # unpool4
            nn.MaxUnpool2d(2, stride=2), )

        # 12*12

        self.deconv_block12 = nn.Sequential(

            # deconv4_1
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv4_2
            nn.ConvTranspose2d(512, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv4_3
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block13 = nn.Sequential(

            # unpool3
            nn.MaxUnpool2d(2, stride=2), )

        # 24*24

        self.deconv_block14 = nn.Sequential(

            # deconv3_1
            nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv3_2
            nn.ConvTranspose2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv3_3
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block15 = nn.Sequential(

            # unpool2
            nn.MaxUnpool2d(2, stride=2), )

        # 48*48

        self.deconv_block16 = nn.Sequential(

            # deconv2_1
            nn.ConvTranspose2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv2_2
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.unpool_block17 = nn.Sequential(

            # unpool1
            nn.MaxUnpool2d(2, stride=2), )

        # 96*96

        self.deconv_block18 = nn.Sequential(

            # deconv1_1
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),

            # deconv1_2
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True), )

        self.seg_score19 = nn.Sequential(

            # seg-score
            nn.Conv2d(64, self.n_classes, 1), )

        if self.learned_billinear:
            raise NotImplementedError

    def forward(self, x):
        size0 = x.size()
        conv1, indices1 = self.conv_block1(x)
        size1 = conv1.size()
        conv2, indices2 = self.conv_block2(conv1)
        size2 = conv2.size()
        conv3, indices3 = self.conv_block3(conv2)
        size3 = conv3.size()
        conv4, indices4 = self.conv_block4(conv3)
        size4 = conv4.size()
        conv5, indices5 = self.conv_block5(conv4)

        conv6 = self.conv_block6(conv5)
        conv7 = self.conv_block7(conv6)
        conv8 = self.deconv_block8(conv7) + conv5
        conv9 = self.unpool(conv8,indices5, output_size=size4)
        conv10 = self.deconv_block10(conv9) + conv4
        conv11 = self.unpool(conv10,indices4, output_size=size3)
        conv12 = self.deconv_block12(conv11) + conv3
        conv13 = self.unpool(conv12,indices3, output_size=size2)
        conv14 = self.deconv_block14(conv13) + conv2
        conv15 = self.unpool(conv14,indices2, output_size=size1)
        conv16 = self.deconv_block16(conv15) + conv1
        conv17 = self.unpool(conv16,indices1, output_size=size0)
        conv18 = self.deconv_block18(conv17)
        out = self.seg_score19(conv18)

        return out

    def init_vgg16_params(self, vgg16, copy_fc8=True):
        blocks = [self.conv_block1,
                  self.conv_block2,
                  self.conv_block3,
                  self.conv_block4,
                  self.conv_block5]

        ranges = [[0, 4], [5, 9], [10, 16], [17, 23], [24, 29]]
        features = list(vgg16.features.children())
        i_layer = 0;
        # copy convolutional filters from vgg16
        for idx, conv_block in enumerate(blocks):
            for l1, l2 in zip(features[ranges[idx][0]:ranges[idx][1]], conv_block):
                if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                    if i_layer == 0:
                        l2.weight.data = ((l1.weight.data[:, 0, :, :] + l1.weight.data[:, 1, :, :] + l1.weight.data[:,
                                                                                                     2, :,
                                                                                                     :]) / 3.0).view(
                            l2.weight.size())
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1
                    else:
                        assert l1.weight.size() == l2.weight.size()
                        assert l1.bias.size() == l2.bias.size()
                        l2.weight.data = l1.weight.data
                        l2.bias.data = l1.bias.data
                        i_layer = i_layer + 1
