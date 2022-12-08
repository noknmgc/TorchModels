import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

import numpy as np

from utils import double_conv


class UnetDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.dconv_center = double_conv(
            512 + hparams["num_dim"], 1024, activation=hparams["activation"]
        )

        self.dconv_up4 = double_conv(1024 + 512, 512, activation=hparams["activation"])
        self.dconv_up3 = double_conv(512 + 256, 256, activation=hparams["activation"])
        self.dconv_up2 = double_conv(256 + 128, 128, activation=hparams["activation"])

        self.dconv_up1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            hparams["activation"],
            nn.Conv2d(64, hparams["out_ch"], kernel_size=3, padding=1),
        )

    def forward(self, x, conv1, conv2, conv3, conv4):
        # Decode
        x = self.dconv_center(x)
        x = self.upsample(x)

        x = self.dconv_up4(torch.cat([x, conv4], dim=1))
        x = self.upsample(x)
        x = self.dconv_up3(torch.cat([x, conv3], dim=1))
        x = self.upsample(x)
        x = self.dconv_up2(torch.cat([x, conv2], dim=1))
        x = self.upsample(x)
        x = self.dconv_up1(torch.cat([x, conv1], dim=1))
        return x

    def decode(self, x, conv1, conv2, conv3, conv4):
        # Decode
        x = self.dconv_center(x)
        x = self.upsample(x)

        x = self.dconv_up4(torch.cat([x, conv4], dim=1))
        x = self.upsample(x)
        x = self.dconv_up3(torch.cat([x, conv3], dim=1))
        x = self.upsample(x)
        x = self.dconv_up2(torch.cat([x, conv2], dim=1))
        x = self.upsample(x)
        x = self.dconv_up1(torch.cat([x, conv1], dim=1))
        return x


class DropOutDecoder(UnetDecoder):
    def __init__(self, hparams, drop_rate=0.3, spatial_dropout=True):
        super(DropOutDecoder, self).__init__(hparams)
        self.dropout = (
            nn.Dropout2d(p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)
        )

    def forward(self, x, conv1, conv2, conv3, conv4):
        x = self.decode(self.dropout(x), conv1, conv2, conv3, conv4)
        return x


class FeatureDropDecoder(UnetDecoder):
    def __init__(self, hparams):
        super(FeatureDropDecoder, self).__init__(hparams)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        return x.mul(drop_mask)

    def forward(self, x, conv1, conv2, conv3, conv4):
        x = self.feature_dropout(x)
        x = self.decode(x, conv1, conv2, conv3, conv4)
        return x


class FeatureNoiseDecoder(UnetDecoder):
    def __init__(self, hparams, uniform_range=0.3):
        super(FeatureNoiseDecoder, self).__init__(hparams)
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x, conv1, conv2, conv3, conv4):
        x = self.feature_based_noise(x)
        x = self.decode(x, conv1, conv2, conv3, conv4)
        return x


def _l2_normalize(d):
    # Normalizing per batch axis
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv(x, decoder, it=1, xi=1e-1, eps=10.0):
    """
    Virtual Adversarial Training
    https://arxiv.org/abs/1704.03976
    """
    x_detached = x.detach()
    with torch.no_grad():
        pred = F.softmax(decoder(x_detached), dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    for _ in range(it):
        d.requires_grad_()
        pred_hat = decoder(x_detached + xi * d)
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction="batchmean")
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return r_adv


if __name__ == "__main__":
    hparams = {"in_ch": 3, "out_ch": 1, "activation": nn.ReLU(inplace=True)}
    decoder = DropOutDecoder(hparams)
