import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform

import numpy as np

from utils import double_conv


class UnetDecoder(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_center = double_conv(
            512 + hparams['num_dim'], 1024, activation=hparams['activation'])

        self.dconv_up4 = double_conv(
            1024 + 512, 512, activation=hparams['activation'])
        self.dconv_up3 = double_conv(
            512 + 256, 256, activation=hparams['activation'])
        self.dconv_up2 = double_conv(
            256 + 128, 128, activation=hparams['activation'])

        self.dconv_up1 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
            hparams['activation'],
            nn.Conv2d(64, hparams['out_ch'], kernel_size=3, padding=1)
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
        self.dropout = nn.Dropout2d(
            p=drop_rate) if spatial_dropout else nn.Dropout(drop_rate)

    def forward(self, x, conv1, conv2, conv3, conv4):
        x = self.decode(self.dropout(x), conv1, conv2, conv3, conv4)
        return x


class FeatureDropDecoder(UnetDecoder):
    def __init__(self, hparams):
        super(FeatureDropDecoder, self).__init__(hparams)

    def feature_dropout(self, x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(
            x.size(0), -1), dim=1, keepdim=True)
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
        noise_vector = self.uni_dist.sample(
            x.shape[1:]).to(x.device).unsqueeze(0)
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
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder.zero_grad()

    r_adv = d * eps
    return r_adv


# class VATDecoder(nn.Module):
#     def __init__(self, upscale, conv_in_ch, num_classes, xi=1e-1, eps=10.0, iterations=1):
#         super(VATDecoder, self).__init__()
#         self.xi = xi
#         self.eps = eps
#         self.it = iterations
#         self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

#     def forward(self, x, _):
#         r_adv = get_r_adv(x, self.upsample, self.it, self.xi, self.eps)
#         x = self.upsample(x + r_adv)
#         return x


# class CutOutDecoder(UnetDecoder):
#     def __init__(self, hparams, erase=0.4):
#         super(CutOutDecoder, self).__init__(hparams)
#         self.erase = erase

#     def forward(self, x, conv1, conv2, conv3, conv4, pred=None):
#         maskcut = self.guided_cutout(
#             pred, erase=self.erase, resize=(x.size(2), x.size(3)))
#         x = x * maskcut
#         x = self.decode(x)
#         return x

#     def guided_cutout(output, upscale, resize, erase=0.4, use_dropout=False):
#         if len(output.shape) == 3:
#             masks = (output > 0).float()
#         else:
#             masks = (output.argmax(1) > 0).float()

#         if use_dropout:
#             p_drop = random.randint(3, 6) / 10
#             maskdroped = (F.dropout(masks, p_drop) > 0).float()
#             maskdroped = maskdroped + (1 - masks)
#             maskdroped.unsqueeze_(0)
#             maskdroped = F.interpolate(maskdroped, size=resize, mode='nearest')

#         masks_np = []
#         for mask in masks:
#             mask_np = np.uint8(mask.cpu().numpy())
#             mask_ones = np.ones_like(mask_np)
#             contours, _ = cv2.findContours(
#                 mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#             polys = [c.reshape(c.shape[0], c.shape[-1])
#                      for c in contours if c.shape[0] > 50]
#             for poly in polys:
#                 min_w, max_w = poly[:, 0].min(), poly[:, 0].max()
#                 min_h, max_h = poly[:, 1].min(), poly[:, 1].max()
#                 bb_w, bb_h = max_w - min_w, max_h - min_h
#                 rnd_start_w = random.randint(0, int(bb_w * (1 - erase)))
#                 rnd_start_h = random.randint(0, int(bb_h * (1 - erase)))
#                 h_start, h_end = min_h + rnd_start_h, min_h + \
#                     rnd_start_h + int(bb_h * erase)
#                 w_start, w_end = min_w + rnd_start_w, min_w + \
#                     rnd_start_w + int(bb_w * erase)
#                 mask_ones[h_start:h_end, w_start:w_end] = 0
#             masks_np.append(mask_ones)
#         masks_np = np.stack(masks_np)

#         maskcut = torch.from_numpy(masks_np).float().unsqueeze_(1)
#         maskcut = F.interpolate(maskcut, size=resize, mode='nearest')

#         if use_dropout:
#             return maskcut.to(output.device), maskdroped.to(output.device)
#         return maskcut.to(output.device)


# class ContextMaskingDecoder(nn.Module):
#     def __init__(self, upscale, conv_in_ch, num_classes):
#         super(ContextMaskingDecoder, self).__init__()
#         self.upscale = upscale
#         self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

#     def forward(self, x, pred=None):
#         x_masked_context = self.guided_masking(x, pred, resize=(x.size(2), x.size(3)),
#                                                upscale=self.upscale, return_msk_context=True)
#         x_masked_context = self.upsample(x_masked_context)
#         return x_masked_context

#     def guided_masking(x, output, upscale, resize, return_msk_context=True):
#         if len(output.shape) == 3:
#             masks_context = (output > 0).float().unsqueeze(1)
#         else:
#             masks_context = (output.argmax(1) > 0).float().unsqueeze(1)

#         masks_context = F.interpolate(
#             masks_context, size=resize, mode='nearest')

#         x_masked_context = masks_context * x
#         if return_msk_context:
#             return x_masked_context

#         masks_objects = (1 - masks_context)
#         x_masked_objects = masks_objects * x
#         return x_masked_objects

# class ObjectMaskingDecoder(nn.Module):
#     def __init__(self, upscale, conv_in_ch, num_classes):
#         super(ObjectMaskingDecoder, self).__init__()
#         self.upscale = upscale
#         self.upsample = upsample(conv_in_ch, num_classes, upscale=upscale)

#     def forward(self, x, pred=None):
#         x_masked_obj = guided_masking(x, pred, resize=(x.size(2), x.size(3)),
#                                       upscale=self.upscale, return_msk_context=False)
#         x_masked_obj = self.upsample(x_masked_obj)

#         return x_masked_obj

if __name__ == '__main__':
    hparams = {'in_ch': 3, 'out_ch': 1, 'activation': nn.ReLU(inplace=True)}
    decoder = DropOutDecoder(hparams)
