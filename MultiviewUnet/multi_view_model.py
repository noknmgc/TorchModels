import torch
import torch.nn as nn


def double_conv(in_channels, out_channels, activation=nn.ReLU(inplace=True)):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        activation,
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        activation,
    )


class Encoder(nn.Module):
    def __init__(
        self, in_ch=3, out_ch=512, activation=nn.ReLU(inplace=True),
    ):
        super().__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.dconv_down1 = double_conv(in_ch, 64, activation=activation)
        self.dconv_down2 = double_conv(64, 128, activation=activation)
        self.dconv_down3 = double_conv(128, 256, activation=activation)
        self.dconv_down4 = double_conv(256, out_ch, activation=activation)

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


class Decoder(nn.Module):
    def __init__(
        self, in_ch=512, out_ch=3, activation=nn.ReLU(inplace=True), last_act=nn.Tanh()
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dconv_up4 = double_conv(in_ch + 512, 512, activation=activation)
        self.dconv_up3 = double_conv(512 + 256, 256, activation=activation)
        self.dconv_up2 = double_conv(256 + 128, 128, activation=activation)
        self.dconv_up1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            activation,
            nn.Conv2d(64, out_ch, kernel_size=3, padding=1),
        )
        self.act = last_act

    def forward(self, x, conv1, conv2, conv3, conv4):
        x = self.dconv_up4(torch.cat([x, conv4], dim=1))
        x = self.upsample(x)
        x = self.dconv_up3(torch.cat([x, conv3], dim=1))
        x = self.upsample(x)
        x = self.dconv_up2(torch.cat([x, conv2], dim=1))
        x = self.upsample(x)
        x = self.dconv_up1(torch.cat([x, conv1], dim=1))
        x = self.act(x)
        return x


class MultiViewUNet(nn.Module):
    def __init__(
        self, in_ch=3, out_ch=3, activation=nn.ReLU(inplace=True), last_act=nn.Tanh()
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.encoder = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.dconv_center = double_conv(512 * 6, 1024, activation=activation)
        self.decoder = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )

    def forward(self, x):
        skip_connect_features = []
        out_features = []
        for i in range(6):
            v = x[:, 3 * i : 3 * i + 3, :, :]
            feature, conv1, conv2, conv3, conv4 = self.encoder(v)
            out_features.append(feature)
            skip_connect_features.append((conv1, conv2, conv3, conv4))

        out_features = torch.cat(out_features, dim=1)
        x = self.dconv_center(out_features)
        x = self.upsample(x)

        outputs = []
        for i in range(6):
            out = self.decoder(x, *skip_connect_features[i])
            outputs.append(out)

        outputs = torch.cat(outputs, dim=1)

        return outputs


class MultiViewUNet2(nn.Module):
    def __init__(
        self, in_ch=3, out_ch=18, activation=nn.ReLU(inplace=True), last_act=nn.Tanh()
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.encoder = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.dconv_center = double_conv(512 * 6, 1024, activation=activation)

        self.conv1_encoder = double_conv(64 * 6, 64, activation=activation)
        self.conv2_encoder = double_conv(128 * 6, 128, activation=activation)
        self.conv3_encoder = double_conv(256 * 6, 256, activation=activation)
        self.conv4_encoder = double_conv(512 * 6, 512, activation=activation)

        self.decoder = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )

    def forward(self, x):
        conv1s, conv2s, conv3s, conv4s = [], [], [], []
        out_features = []
        for i in range(6):
            v = x[:, 3 * i : 3 * i + 3, :, :]
            feature, conv1, conv2, conv3, conv4 = self.encoder(v)
            out_features.append(feature)
            conv1s.append(conv1)
            conv2s.append(conv2)
            conv3s.append(conv3)
            conv4s.append(conv4)

        out_features = torch.cat(out_features, dim=1)
        conv1s = self.conv1_encoder(torch.cat(conv1s, dim=1))
        conv2s = self.conv2_encoder(torch.cat(conv2s, dim=1))
        conv3s = self.conv3_encoder(torch.cat(conv3s, dim=1))
        conv4s = self.conv4_encoder(torch.cat(conv4s, dim=1))

        x = self.dconv_center(out_features)
        x = self.upsample(x)
        out = self.decoder(x, conv1s, conv2s, conv3s, conv4s)

        return out


class MultiViewSeparateUNet(nn.Module):
    def __init__(
        self, in_ch=3, out_ch=3, activation=nn.ReLU(inplace=True), last_act=nn.Tanh()
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.encoder1 = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.encoder2 = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.encoder3 = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.encoder4 = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.encoder5 = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.encoder6 = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.dconv_center = double_conv(512 * 6, 1024, activation=activation)
        # self.dconv_center2 = double_conv(2048, 1024, activation=activation)
        self.decoder1 = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )
        self.decoder2 = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )
        self.decoder3 = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )
        self.decoder4 = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )
        self.decoder5 = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )
        self.decoder6 = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )
        self.encoders = [
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
            self.encoder5,
            self.encoder6,
        ]
        self.decoders = [
            self.decoder1,
            self.decoder2,
            self.decoder3,
            self.decoder4,
            self.decoder5,
            self.decoder6,
        ]

    def forward(self, x):
        skip_connect_features = []
        out_features = []
        for i in range(6):
            v = x[:, 3 * i : 3 * i + 3, :, :]
            feature, conv1, conv2, conv3, conv4 = self.encoders[i](v)
            out_features.append(feature)
            skip_connect_features.append((conv1, conv2, conv3, conv4))

        out_features = torch.cat(out_features, dim=1)
        x = self.dconv_center(out_features)
        # x = self.dconv_center2(x)
        x = self.upsample(x)

        outputs = []
        for i in range(6):
            out = self.decoders[i](x, *skip_connect_features[i])
            outputs.append(out)

        outputs = torch.cat(outputs, dim=1)

        return outputs


class MultiViewSeparateUNet_gray(nn.Module):
    def __init__(
        self, in_ch=1, out_ch=1, activation=nn.ReLU(inplace=True), last_act=nn.Tanh()
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.encoder1 = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.encoder2 = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.encoder3 = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.encoder4 = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.encoder5 = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.encoder6 = Encoder(in_ch=in_ch, out_ch=512, activation=activation)
        self.dconv_center = double_conv(512 * 6, 1024, activation=activation)
        self.decoder1 = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )
        self.decoder2 = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )
        self.decoder3 = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )
        self.decoder4 = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )
        self.decoder5 = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )
        self.decoder6 = Decoder(
            in_ch=1024, out_ch=out_ch, activation=activation, last_act=last_act,
        )
        self.encoders = [
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
            self.encoder5,
            self.encoder6,
        ]
        self.decoders = [
            self.decoder1,
            self.decoder2,
            self.decoder3,
            self.decoder4,
            self.decoder5,
            self.decoder6,
        ]

    def forward(self, x):
        skip_connect_features = []
        out_features = []
        for i in range(6):
            v = x[:, i : i + 1, :, :]
            feature, conv1, conv2, conv3, conv4 = self.encoders[i](v)
            out_features.append(feature)
            skip_connect_features.append((conv1, conv2, conv3, conv4))

        out_features = torch.cat(out_features, dim=1)
        x = self.dconv_center(out_features)
        x = self.upsample(x)

        outputs = []
        for i in range(6):
            out = self.decoders[i](x, *skip_connect_features[i])
            outputs.append(out)

        outputs = torch.cat(outputs, dim=1)

        return outputs


if __name__ == "__main__":
    net = MultiViewSeparateUNet(
        in_ch=3, out_ch=3, activation=nn.ReLU(inplace=True), last_act=nn.Tanh()
    )
    input = torch.ones(4, 3 * 6, 256, 256)
    out = net(input)
    print(out.shape)
