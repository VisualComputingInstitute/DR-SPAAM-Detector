import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
# From https://github.com/mbsariyildiz/focal-loss.pytorch/blob/master/focalloss.py
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)

    def forward(self, input, target, reduction='mean'):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt

        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'none':
            return loss
        else:
            raise RuntimeError


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=-1):
        super(BinaryFocalLoss, self).__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, pred, target, reduction='mean'):
        return binary_focal_loss(pred, target, self.gamma, self.alpha, reduction)


def binary_focal_loss(pred, target, gamma=2.0, alpha=-1, reduction='mean'):
    loss_pos = - target * (1.0 - pred)**gamma * torch.log(pred)
    loss_neg = - (1.0 - target) * pred**gamma * torch.log(1.0 - pred)

    if alpha >= 0.0 and alpha <= 1.0:
        loss_pos = loss_pos * alpha
        loss_neg = loss_neg * (1.0 - alpha)

    loss = loss_pos + loss_neg

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise RuntimeError