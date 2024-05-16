import copy
from email.policy import default
from enum import Enum
import torch
import argparse
from torch_geometric import data
from torch_geometric.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F
from wilds.common.utils import split_into_groups


def get_irm_loss(causal_pred, labels, batch_env_idx, criterion=F.cross_entropy):
    device = causal_pred.device
    dummy_w = torch.tensor(1.).to(device).requires_grad_()
    loss_0 = criterion(causal_pred[batch_env_idx == 0] * dummy_w, labels[batch_env_idx == 0])
    loss_1 = criterion(causal_pred[batch_env_idx == 1] * dummy_w, labels[batch_env_idx == 1])
    grad_0 = torch.autograd.grad(loss_0, dummy_w, create_graph=True)[0]
    grad_1 = torch.autograd.grad(loss_1, dummy_w, create_graph=True)[0]
    irm_loss = torch.sum(grad_0 * grad_1)

    return irm_loss

def get_irm_loss_env(causal_pred, labels, batch_env_idx, criterion=F.cross_entropy):
    # constructed by enhwa 2023-06-13
    # from https://github.com/facebookresearch/InvariantRiskMinimization/blob/main/code/colored_mnist/main.py
    device = causal_pred.device
    dummy_w = torch.tensor(1.).to(device).requires_grad_()
    _, group_indices, _ = split_into_groups(batch_env_idx)
    grads = []
    for i_group in group_indices:
        group_losses = criterion(causal_pred[i_group] * dummy_w, labels[i_group])
        grad = torch.autograd.grad(group_losses, dummy_w, create_graph=True)[0]
        grads.append(torch.sum(grad ** 2))
    irm_loss = torch.stack(grads).mean()
    return irm_loss

def get_contrast_loss(causal_rep, labels, norm=None, contrast_t=1.0, sampling='mul', y_pred=None):

    if norm != None:
        causal_rep = F.normalize(causal_rep)
    if sampling.lower() in ['mul', 'var']:
        # modified from https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11
        device = causal_rep.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask: no need
        # mask = mask.repeat(anchor_count, contrast_count)
        batch_size = labels.size(0)
        anchor_count = 1
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # print(log_prob)
        # print(mask.sum(1))
        # compute mean of log-likelihood over positive
        is_valid = mask.sum(1) != 0
        mean_log_prob_pos = (mask * log_prob).sum(1)[is_valid] / mask.sum(1)[is_valid]
        # some classes may not be sampled by more than 2
        # mean_log_prob_pos[torch.isnan(mean_log_prob_pos)] = 0.0

        # loss
        # contrast_loss = -(args.temperature / args.base_temperature) * mean_log_prob_pos
        # contrast_loss = contrast_loss.view(anchor_count, batch_size).mean()
        contrast_loss = -mean_log_prob_pos.mean()
        if sampling.lower() == 'var':
            contrast_loss += mean_log_prob_pos.var()
    elif sampling.lower() == 'single':
        N = causal_rep.size(0)
        pos_idx = torch.arange(N)
        neg_idx = torch.randperm(N)
        for i in range(N):
            for j in range(N):
                if labels[i] == labels[j]:
                    pos_idx[i] = j
                else:
                    neg_idx[i] = j
        contrast_loss = -torch.mean(
            torch.bmm(causal_rep.unsqueeze(1), causal_rep[pos_idx].unsqueeze(1).transpose(1, 2)) -
            torch.matmul(causal_rep.unsqueeze(1), causal_rep[neg_idx].unsqueeze(1).transpose(1, 2)))
    elif sampling.lower() == 'cncp':
        # correct & contrast with hard postive only https://arxiv.org/abs/2203.01517
        device = causal_rep.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask: no need
        # mask = mask.repeat(anchor_count, contrast_count)
        batch_size = labels.size(0)
        anchor_count = 1
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        # find hard postive & negative
        pos_mask = y_pred != labels
        neg_mask = y_pred == labels

        # hard negative: diff label && correct pred
        neg_mask = torch.logical_not(mask)  #* neg_mask
        # hard positive: same label && incorrect pred
        pos_mask = mask * pos_mask

        # compute log_prob
        neg_exp_logits = torch.exp(logits) * neg_mask
        pos_exp_logits = torch.exp(logits) * pos_mask
        log_prob = logits - \
                    torch.log(pos_exp_logits.sum(1, keepdim=True) + \
                            neg_exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        is_valid = pos_mask.sum(1) != 0
        mean_log_prob_pos = (pos_mask * log_prob).sum(1)[is_valid] / pos_mask.sum(1)[is_valid]
        # some classes may not be sampled by more than 2
        # mean_log_prob_pos[torch.isnan(mean_log_prob_pos)] = 0.0

        # loss
        # contrast_loss = -(args.temperature / args.base_temperature) * mean_log_prob_pos
        # contrast_loss = contrast_loss.view(anchor_count, batch_size).mean()
        contrast_loss = -mean_log_prob_pos.mean()
    elif sampling.lower() == 'cnc':
        # correct & contrast https://arxiv.org/abs/2203.01517
        device = causal_rep.device
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(causal_rep, causal_rep.T), contrast_t)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # tile mask: no need
        # mask = mask.repeat(anchor_count, contrast_count)
        batch_size = labels.size(0)
        anchor_count = 1
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        # find hard postive & negative
        pos_mask = y_pred != labels
        neg_mask = y_pred == labels
        # hard negative: diff label && correct pred
        neg_mask = torch.logical_not(mask) * neg_mask * logits_mask
        # hard positive: same label && incorrect pred
        pos_mask = mask * pos_mask
        if neg_mask.sum() == 0:
            neg_mask = torch.logical_not(mask)
        if pos_mask.sum() == 0:
            pos_mask = mask
        # compute log_prob
        neg_exp_logits = torch.exp(logits) * neg_mask
        pos_exp_logits = torch.exp(logits) * pos_mask
        log_prob = logits - \
                    torch.log(pos_exp_logits.sum(1, keepdim=True) + \
                            neg_exp_logits.sum(1, keepdim=True)+1e-12)
        # compute mean of log-likelihood over positive
        is_valid = pos_mask.sum(1) != 0
        mean_log_prob_pos = (pos_mask * log_prob).sum(1)[is_valid] / pos_mask.sum(1)[is_valid]
        contrast_loss = -mean_log_prob_pos.mean()
    else:
        raise Exception("Not implmented contrasting method")
    return contrast_loss




def KLDist(p, q, eps=1e-8):
    log_p, log_q = torch.log(p + eps), torch.log(q + eps)
    return torch.sum(p * (log_p - log_q.to(p.device)))


def bce_log(pred, gt, eps=1e-8):
    prob = torch.sigmoid(pred)
    return -(gt * torch.log(prob + eps) + (1 - gt) * torch.log(1 - prob + eps))


class MeanLoss(torch.nn.Module):
    def __init__(self, base_loss):
        super(MeanLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, pred, gt, domain):
        _, group_indices, _ = split_into_groups(domain)
        total_loss, total_cnt = 0, 0
        for i_group in group_indices:
            total_loss += self.base_loss(pred[i_group], gt[i_group])
            total_cnt += 1
        return total_loss / total_cnt


class DeviationLoss(torch.nn.Module):
    def __init__(self, activation, reduction='mean'):
        super(DeviationLoss, self).__init__()
        assert activation in ['relu', 'abs', 'none'],\
            'Invaild activation function'
        assert reduction in ['mean', 'sum'], \
            'Invalid reduction method'

        self.activation = activation
        self.reduction = reduction

    def forward(self, pred, condition_pred_mean):
        if self.activation == 'relu':
            loss = torch.relu(pred - condition_pred_mean)
        elif self.activation == 'abs':
            loss = torch.abs(pred - condition_pred_mean)
        else:
            loss = pred - condition_pred_mean

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)

from torch.distributions.normal import Normal

def discrete_gaussian(nums, std=1):
    Dist = Normal(loc=0, scale=1)
    plen, halflen = std * 6 / nums, std * 3 / nums
    posx = torch.arange(-3 * std + halflen, 3 * std, plen)
    result = Dist.cdf(posx + halflen) - Dist.cdf(posx - halflen)
    return result / result.sum()
