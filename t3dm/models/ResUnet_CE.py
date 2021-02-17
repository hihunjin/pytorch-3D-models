# based on V-Net
# https://arxiv.org/abs/1904.00592

import torch
import torch.nn as nn
import torch.nn.functional as F

dropout_rate = 0.3
num_organ = 1
size = 48


# Define a single 3D FCN
class ResUNet(nn.Module):
    def __init__(self, training, inchannel, stage):
        """
        :param training     : Does the logo network belong to the training stage or the testing stage?
        :param inchannel    : Number of input channels at the start of the network
        :param stage        : Mark networking is in phase one or phase two.
        """
        super().__init__()

        self.training = training
        self.stage = stage

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(inchannel, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),

            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),

            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),

            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),

            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        self.map = nn.Conv3d(32, num_organ + 1, 1)

    def forward(self, inputs):

        if self.stage is 'stage1':
            long_range1 = self.encoder_stage1(inputs) + inputs
        else:
            long_range1 = self.encoder_stage1(inputs)

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, dropout_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, dropout_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, dropout_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, dropout_rate, self.training)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, dropout_rate, self.training)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, dropout_rate, self.training)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        outputs = self.map(outputs)

        # return probability plot
        return outputs


# Defining the final series, 3-D FCN
class Net(nn.Module):
    def __init__(self, training, z_len = size):
        super().__init__()

        self.training = training

        self.stage1 = ResUNet(training=training, inchannel=1, stage='stage1')
        self.stage2 = ResUNet(training=training, inchannel=num_organ + 2, stage='stage2')

        self.z_len = z_len

    def forward(self, inputs):
        """
        Double the input data on the axis up and into the first stage network
        get the result of a division at a rough scale
        The raw scale data are then spliced with the segmentation results obtained in the first step and fed into the second stage network
        get the final segmentation result
        """

        inputs_dim = inputs.size(-1)
        # First, double the input.
        if inputs.size()[2:] != [self.z_len,inputs_dim//2,inputs_dim//2]:
            inputs_stage1 = F.interpolate(inputs, (self.z_len, inputs_dim//2, inputs_dim//2), mode='trilinear', align_corners=True)

        # get the first-stage result
        output_stage1 = self.stage1(inputs_stage1)
        if inputs.size()[2:] != [self.z_len,inputs_dim,inputs_dim]:
            # output_stage1 = F.interpolate(output_stage1, (size, 256, 256), mode='trilinear', align_corners=True)      #원래 있던 것.
            output_stage1 = nn.Upsample((self.z_len, inputs_dim, inputs_dim), mode='trilinear', align_corners=True)(output_stage1)
        # print('Upsample',output_stage1.size());import sys;sys.exit()

        temp = F.softmax(output_stage1, dim=1)

        # Patch the first stage results with the original input data as the second stage input
        inputs_stage2 = torch.cat((temp, inputs), dim=1)

        # get the second stage result
        output_stage2 = self.stage2(inputs_stage2)

        if self.training is True:
            return output_stage1, output_stage2         #size : stage1,stage2:[1,2,128,256,256]
        else:
            return output_stage2


# network parameter initialization function
def init(module):
    if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(module.weight.data, 0.25)
        nn.init.constant_(module.bias.data, 0)


net = Net(training=True)
net.apply(init)

# # output data dimension check
# net = net.cuda()
# data = torch.randn((1, 1, 48, 256, 256)).cuda()
#
# with torch.no_grad():
#     res = net(data)
#
# for item in res:
#     print(item.size())
#
# # 计算网络参数
# num_parameter = .0
# for item in net.modules():
#
#     if isinstance(item, nn.Conv3d) or isinstance(item, nn.ConvTranspose3d):
#         num_parameter += (item.weight.size(0) * item.weight.size(1) *
#                           item.weight.size(2) * item.weight.size(3) * item.weight.size(4))
#
#         if item.bias is not None:
#             num_parameter += item.bias.size(0)
#
#     elif isinstance(item, nn.PReLU):
#         num_parameter += item.num_parameters
#
#
# print(num_parameter)


if __name__=='__main__':
    from torchsummary import summary
    z_len = 40
    net = Net(training=True, z_len = z_len)
    with torch.no_grad():
        summary(net.cuda(), (1,z_len,512,512),device='cuda')
