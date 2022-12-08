import torch.nn as nn

from utils import double_conv


class UnetEncoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)

        self.dconv_down1 = double_conv(
            hparams["in_ch"], 64, activation=hparams["activation"]
        )
        self.dconv_down2 = double_conv(64, 128, activation=hparams["activation"])
        self.dconv_down3 = double_conv(128, 256, activation=hparams["activation"])
        self.dconv_down4 = double_conv(256, 512, activation=hparams["activation"])

    def forward(self, x):
        # Encoder
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        return x, conv1, conv2, conv3, conv4
