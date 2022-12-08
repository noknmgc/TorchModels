import torch
import torch.nn as nn

from encoder import UnetEncoder
from decoder import UnetDecoder, DropOutDecoder, FeatureDropDecoder, FeatureNoiseDecoder


class Unet(nn.Module):
    def __init__(
        self,
        hparams={
            "in_ch": 3,
            "out_ch": 1,
            "num_dim": 0,
            "activation": nn.ReLU(inplace=True),
        },
    ):
        super().__init__()

        self.encoder = UnetEncoder(hparams)
        self.main_decoder = UnetDecoder(hparams)

        drop_decoder = [DropOutDecoder(hparams)]
        feature_drop = [FeatureDropDecoder(hparams)]
        feature_noise = [FeatureNoiseDecoder(hparams)]

        self.aux_decoders = nn.ModuleList(
            [*drop_decoder, *feature_drop, *feature_noise]
        )

        self.act = nn.Hardsigmoid()

    def forward(self, x, unlabeled=False):
        # Encoder
        x, *conv1_4 = self.encoder(x)
        # Decoder
        out_main = self.main_decoder(x, *conv1_4)
        out_main = self.act(out_main)

        if not unlabeled:
            return out_main

        outputs = [self.act(decoder(x, *conv1_4)) for decoder in self.aux_decoders]

        return out_main, outputs


class Unet_CCT(nn.Module):
    def __init__(
        self,
        hparams={
            "in_ch": 3,
            "out_ch": 1,
            "num_dim": 1,
            "activation": nn.ReLU(inplace=True),
        },
    ):
        super().__init__()

        self.encoder = UnetEncoder(hparams)
        self.main_decoder = UnetDecoder(hparams)

        drop_decoder = [DropOutDecoder(hparams)]
        feature_drop = [FeatureDropDecoder(hparams)]
        feature_noise = [FeatureNoiseDecoder(hparams)]

        self.aux_decoders = nn.ModuleList(
            [*drop_decoder, *feature_drop, *feature_noise]
        )

        self.act = nn.Hardsigmoid()

    def forward(self, x, num, unlabeled=False):
        # Encoder
        x, *conv1_4 = self.encoder(x)
        # Decoder
        num = torch.einsum(
            "ij,iklm->iklm",
            num,
            torch.ones(x.size(0), 1, x.size(2), x.size(3), device=num.device),
        )
        x = torch.cat([x, num], dim=1)
        out_main = self.main_decoder(x, *conv1_4)
        out_main = self.act(out_main)

        if not unlabeled:
            return out_main

        outputs = [self.act(decoder(x, *conv1_4)) for decoder in self.aux_decoders]

        return out_main, outputs


if __name__ == "__main__":
    hparams = {
        "in_ch": 3,
        "out_ch": 1,
        "num_dim": 0,
        "activation": nn.ReLU(inplace=True),
    }
    net = Unet(hparams=hparams)
    input = torch.ones(4, 3, 64, 64)
    out = net(input)
    print(out.shape)

    out, outs = net(input, unlabeled=True)
    print(out.shape)
    for i in outs:
        print(i.shape)

    print("---------")
    hparams = {
        "in_ch": 3,
        "out_ch": 1,
        "num_dim": 1,
        "activation": nn.ReLU(inplace=True),
    }
    net = Unet_CCT(hparams=hparams)
    input = (torch.ones(4, 3, 64, 64), torch.ones(4, 1))
    out = net(*input)
    print(out.shape)

    out, outs = net(*input, unlabeled=True)
    print(out.shape)
    for i in outs:
        print(i.shape)
