from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss_utils import FocalLoss, BinaryFocalLoss


def _conv(in_channel, out_channel, kernel_size, padding):
    return nn.Sequential(nn.Conv1d(in_channel, out_channel,
                                   kernel_size=kernel_size, padding=padding),
                         nn.BatchNorm1d(out_channel),
                         nn.LeakyReLU(negative_slope=0.1, inplace=True))


def _conv3x3(in_channel, out_channel):
    return _conv(in_channel, out_channel, kernel_size=3, padding=1)


def _conv1x1(in_channel, out_channel):
    return _conv(in_channel, out_channel, kernel_size=1, padding=1)


class DROW(nn.Module):
    def __init__(self, dropout=0.5, num_scans=5, num_pts=48, focal_loss_gamma=0.0,
                 pedestrian_only=False):
        super(DROW, self).__init__()

        self.dropout = dropout

        self.conv_block_1 = nn.Sequential(_conv3x3(1, 64),
                                          _conv3x3(64, 64),
                                          _conv3x3(64, 128))
        self.conv_block_2 = nn.Sequential(_conv3x3(128, 128),
                                          _conv3x3(128, 128),
                                          _conv3x3(128, 256))
        self.conv_block_3 = nn.Sequential(_conv3x3(256, 256),
                                          _conv3x3(256, 256),
                                          _conv3x3(256, 512))
        self.conv_block_4 = nn.Sequential(_conv3x3(512, 256),
                                          _conv3x3(256, 128))

        if pedestrian_only:
            self.conv_cls = nn.Conv1d(128, 1, kernel_size=1)  # probs
            self.cls_loss = BinaryFocalLoss(gamma=focal_loss_gamma) \
                    if focal_loss_gamma > 0.0 else F.binary_cross_entropy
        else:
            self.conv_cls = nn.Conv1d(128, 4, kernel_size=1)  # probs
            self.cls_loss = FocalLoss(gamma=focal_loss_gamma) \
                    if focal_loss_gamma > 0.0 else F.cross_entropy

        self.conv_reg = nn.Conv1d(128, 2, kernel_size=1)  # vote

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_conv(self, x, conv_block):
        out = conv_block(x)
        out = F.max_pool1d(out, kernel_size=2)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)

        return out

    def _forward_cutout(self, x):
        n_batch, n_cutout, n_scan, n_pts = x.shape

        out = x.view(n_batch * n_cutout * n_scan, 1, n_pts)

        # feature for each cutout
        out = self._forward_conv(out, self.conv_block_1)  # 24
        out = self._forward_conv(out, self.conv_block_2)  # 12

        # (batch, cutout, scan, channel, pts)
        return out.view(n_batch, n_cutout, n_scan, out.shape[-2], out.shape[-1])

    def _fuse_cutout(self, x):
        return torch.sum(x, dim=2)  # (batch, cutout, channel, pts)

    def _forward_fused_cutout(self, x):
        n_batch, n_cutout, n_channel, n_pts = x.shape

        # feature for fused cutout
        out = x.view(n_batch*n_cutout, n_channel, n_pts)
        out = self._forward_conv(out, self.conv_block_3)  # 6
        out = self.conv_block_4(out)
        out = F.avg_pool1d(out, kernel_size=out.shape[-1])  # (batch*cutout, channel, 1)

        pred_cls = self.conv_cls(out).view(n_batch, n_cutout, -1)
        pred_reg = self.conv_reg(out).view(n_batch, n_cutout, 2)

        return pred_cls, pred_reg

    def forward(self, x):
        out = self._forward_cutout(x)
        out = self._fuse_cutout(out)
        pred_cls, pred_reg = self._forward_fused_cutout(out)

        return pred_cls, pred_reg


class _TemporalAttention(nn.Module):
    def __init__(self, n_scans, n_pts, n_channel):
        super(_TemporalAttention, self).__init__()
        self.conv1 = nn.Conv1d(n_channel, 128, kernel_size=n_pts, padding=0)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=n_scans, padding=0)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, n_scans)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        n_batch, n_scans, n_channel, n_pts = x.shape

        out = x.view(n_batch * n_scans, n_channel, n_pts)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = out.view(n_batch, n_scans, 128).permute(0, 2, 1)  # (batch, feature, scans)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out).view(n_batch, 64)  # (batch, feature)

        out = self.fc(out)
        out = F.softmax(out, dim=1)  # (batch, scans)

        return out


class TemporalDROW(DROW):
    def __init__(self, dropout=0.5, num_scans=5, num_pts=48, focal_loss_gamma=0.0,
                 pedestrian_only=False):
        super(TemporalDROW, self).__init__(
                dropout=dropout, num_scans=num_scans, num_pts=num_pts,
                focal_loss_gamma=focal_loss_gamma, pedestrian_only=pedestrian_only)

        if num_scans > 1:
            self.gate = _TemporalAttention(num_scans, ceil(num_pts / 4), 256)

    def _fuse_cutout(self, x):
        n_batch, n_cutout, n_scans, n_channel, n_pts = x.shape

        if n_scans == 1:
            return x.view(n_batch, n_cutout, n_channel, n_pts)

        out = x.view(n_batch * n_cutout, n_scans, n_channel, n_pts)
        gate = self.gate(out)
        out = out * gate[..., None, None]
        out = torch.sum(out, dim=1)    # (batch*cutout, channel, pts)

        return out.view(n_batch, n_cutout, n_channel, n_pts)

    def forward(self, x, testing=False, fea_prev=None):
        # inference
        if testing:
            out = self._forward_cutout(x).squeeze(dim=2)
            fea_now = out.clone()
            if fea_prev is not None and len(fea_prev) > 0:
                out = torch.stack(list(fea_prev) + [out], dim=2)
                out = self._fuse_cutout(out)
            pred_cls, pred_reg = self._forward_fused_cutout(out)

            return pred_cls, pred_reg, fea_now

        out = self._forward_cutout(x)
        out = self._fuse_cutout(out)
        pred_cls, pred_reg = self._forward_fused_cutout(out)

        return pred_cls, pred_reg


class _SpatialAttention(nn.Module):
    def __init__(self, n_pts, n_channel, alpha=0.5, window_size=7):
        super(_SpatialAttention, self).__init__()
        self._alpha = alpha
        self._window_size = window_size

        self.conv = nn.Sequential(
                nn.Conv1d(n_channel, 128, kernel_size=n_pts, padding=0),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # place holder, created at runtime
        self.neighbor_masks, self.neighbor_inds = None, None

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _generate_neighbor_mask(self, x):
        # indices of neighboring cutout
        n_cutout = x.shape[1]
        hw = int(self._window_size / 2)
        inds_col = torch.arange(n_cutout).unsqueeze(dim=-1).long()
        window_inds = torch.arange(-hw, hw+1).long()
        inds_col = inds_col + window_inds.unsqueeze(dim=0)  # (cutout, neighbors)
        inds_col = inds_col.clamp(min=0, max=n_cutout-1)
        inds_row = torch.arange(n_cutout).unsqueeze(dim=-1).expand_as(inds_col).long()
        inds_full = torch.stack((inds_row, inds_col), dim=2).view(-1, 2)
        # self.register_buffer('neighbor_inds', inds_full)

        masks = torch.zeros(n_cutout, n_cutout).float()
        masks[inds_full[:, 0], inds_full[:, 1]] = 1.0
        return masks.cuda(x.get_device()) if x.is_cuda else masks, inds_full

    def forward(self, x, x_template):
        n_batch, n_cutout, n_channel, n_pts = x.shape

        # # for ablation study - no spatial attention
        # if True:
        #     out_temp = self._alpha * x + (1.0 - self._alpha) * x_template
        #     return out_temp, None

        # only need to generate neighbor mask once
        if self.neighbor_masks is None:
            self.neighbor_masks, self.neighbor_inds = self._generate_neighbor_mask(x)

        # embedding for cutout
        emb_x = self.conv(x.view(n_batch * n_cutout, n_channel, n_pts))
        emb_x = emb_x.view(n_batch, n_cutout, 128)

        # embedding for template
        emb_temp = self.conv(x_template.view(n_batch * n_cutout, n_channel, n_pts))
        emb_temp = emb_temp.view(n_batch, n_cutout, 128)

        # pair-wise similarity (batch, cutout, cutout)
        sim = torch.matmul(emb_x, emb_temp.permute(0, 2, 1))

        # # masked softmax (original)
        # # @note 1e-5 was added to `exps` before, not to `exps_sum`
        # maxes = (sim * self.neighbor_masks).max(dim=-1, keepdim=True)[0]
        # sim_centered = torch.clamp(sim - maxes, max=0.0)
        # exps = torch.exp(sim_centered) * self.neighbor_masks
        # exps_sum = exps.sum(dim=-1, keepdim=True)
        # sim = exps / exps_sum

        # masked softmax (new)
        sim = sim - 1e10 * (1.0 - self.neighbor_masks)  # make sure the out-of-window elements have small values
        maxes = sim.max(dim=-1, keepdim=True)[0]
        exps = torch.exp(sim - maxes) * self.neighbor_masks
        exps_sum = exps.sum(dim=-1, keepdim=True)
        sim = exps / exps_sum

#        # weighted average on the template (old)
#        out_temp = x_template.view(n_batch, n_cutout, n_channel*n_pts).permute(0, 2, 1)
#        out_temp = torch.matmul(out_temp, sim.permute(0, 2, 1))
#        out_temp = out_temp.permute(0, 2, 1).view(
#                n_batch, n_cutout, n_channel, n_pts)

        # weighted average on the template (new, remove redundent transpose)
        out_temp = x_template.view(n_batch, n_cutout, n_channel*n_pts)
        out_temp = torch.matmul(sim, out_temp)
        out_temp = out_temp.view(n_batch, n_cutout, n_channel, n_pts)

        # auto-regressive
        out_temp = self._alpha * x + (1.0 - self._alpha) * out_temp

        return out_temp, sim


class SpatialDROW(DROW):
    def __init__(self, dropout=0.5, num_scans=5, num_pts=48, focal_loss_gamma=0.0,
                 alpha=0.5, window_size=7, pedestrian_only=False):
        super(SpatialDROW, self).__init__(
                dropout=dropout, num_scans=num_scans, num_pts=num_pts,
                focal_loss_gamma=focal_loss_gamma, pedestrian_only=pedestrian_only)

        self.gate = _SpatialAttention(n_pts=int(ceil(num_pts / 4)),
                                      n_channel=256,
                                      alpha=alpha,
                                      window_size=window_size)

    def forward(self, x, testing=False, fea_template=None):
        # inference
        if testing:
            out = self._forward_cutout(x).squeeze(dim=2)
            if fea_template is None:
                out_template = out.clone()
                sim = None
            else:
                out_template, sim = self.gate(out, fea_template)

            pred_cls, pred_reg = self._forward_fused_cutout(out_template)

            return pred_cls, pred_reg, out_template, sim

        # # for ablation study - no auto-regression
        # if True:
        #     input = x[:, :, -2, :].unsqueeze(dim=2)
        #     out_template = self._forward_cutout(input).squeeze(dim=2)
        #     input = x[:, :, -1, :].unsqueeze(dim=2)
        #     out = self._forward_cutout(input).squeeze(dim=2)
        #     out_template, sim = self.gate(out, out_template)
        #     pred_cls, pred_reg = self._forward_fused_cutout(out_template)
        #     return pred_cls, pred_reg, sim

        # training or evaluation
        n_scan = x.shape[2]
        input = x[:, :, 0, :].unsqueeze(dim=2)
        out_template = self._forward_cutout(input).squeeze(dim=2)
        for i in range(1, n_scan):
            input = x[:, :, i, :].unsqueeze(dim=2)
            out = self._forward_cutout(input).squeeze(dim=2)
            out_template, sim = self.gate(out, out_template)

        pred_cls, pred_reg = self._forward_fused_cutout(out_template)

        return pred_cls, pred_reg, sim
