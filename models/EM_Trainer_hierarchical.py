import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.autograd import grad
from utils.util import args_print, set_seed, eval_model
from copy import deepcopy
from wilds.common.utils import split_into_groups
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


class EM_EDNIL_Trainer_IL_hier:
    def __init__(self, num_classes, model, optimizer, device, args):
        self.erm_criterion = nn.CrossEntropyLoss(reduction='none')
        self.num_classes = num_classes
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.batch_size = args.batch_size
        self.irm_opt = args.irm_opt
        self.il_last_hierarchy = args.il_last_hierarchy
        if 'erm' in self.irm_opt:
            self.irm_p = -1
        else:
            self.irm_p = args.irm_p
        self.l2_w = args.l2_w
        if 'vrex' in args.irm_opt:
            self._penalty = self._v_rex_penalty
        else:
            self._penalty = self._irm_penalty

    def _irm_penalty(self, preds, y, batch_env_idx):
        scale = torch.tensor(1.).to(self.device).requires_grad_()
        _, group_indices, _ = split_into_groups(batch_env_idx)            
        grads = []
        for i_group in group_indices:            
            erm_loss = self.erm_criterion(preds[i_group] * scale, y[i_group]).mean()
            grad_single = grad(erm_loss, [scale], create_graph=True)[0]
            grads.append(grad_single.pow(2).sum())
        irm_loss = torch.stack(grads).mean()
        return irm_loss

    def _v_rex_penalty(self, preds, y, batch_env_idx):
        _, group_indices, _ = split_into_groups(batch_env_idx)
        group_losses = []
        for i_group in group_indices:
            erm_loss = self.erm_criterion(preds[i_group], y[i_group]).mean()
            group_losses.append(erm_loss)
        irm_loss = torch.var(torch.FloatTensor(group_losses).to(self.device))
        return irm_loss

    def compute_il_loss(self, data_loader, env_model=None):
        def _model_weight_norm():
            weight_norm = torch.tensor(0.).to(self.device)
            for w in self.model.parameters():
                weight_norm += w.norm().pow(2)
            return weight_norm

        erm_loss_avg = 0.
        penalty_avg = 0.
        loss_avg = 0.
        
        env_infer_ids = []
        
        for step, graph in enumerate(data_loader):
            graph = graph.to(self.device)
            graph_ = graph
            if 'EI' in self.irm_opt and self.il_last_hierarchy:
                edge_mask = torch.full((graph.edge_index.shape[1],), -1, device=graph.edge_index.device)
                for i in range(env_model.num_hieararchy):
                    domain_preds, preds, graph, c_rep, s_rep, edge_mask = env_model(graph, i, edge_mask)
                env_idx = domain_preds.argmax(dim=1)
            elif 'random' in self.irm_opt:
                env_idx = torch.randint(0, 2, (len(graph.y),)).to(self.device)
            elif 'group' in self.irm_opt:
                env_idx = graph.group
            elif 'erm' in self.irm_opt:
                env_idx = None
            else:
                raise NotImplementedError

            preds, rep = self.model(graph_, env_model)

            if len(graph.y.size()) == 2:
                y = graph.y.squeeze(1)
            else:
                y = graph.y
            
            erm_loss = self.erm_criterion(preds, y).mean()
            
            if env_idx is not None:
                penalty = self._penalty(preds, y, env_idx)
            else:
                penalty = torch.tensor(0.).to(self.device)

            if self.l2_w > 0:
                weight_norm = _model_weight_norm()
            else:
                weight_norm = 0                


            if self.irm_p > 0:
                batch_loss = erm_loss + self.irm_p * penalty + self.l2_w * weight_norm
            else:
                batch_loss = erm_loss + self.l2_w * weight_norm

            erm_loss_avg += erm_loss.item()
            penalty_avg += self.irm_p * penalty.item()
            loss_avg += batch_loss.item()
            
            self.optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()            
            
        erm_loss_avg /= len(data_loader)
        penalty_avg /= len(data_loader)
        loss_avg /= len(data_loader)
        
        return loss_avg, erm_loss_avg, penalty_avg

    def train_IL_hier(self, train_loader, valid_loader, test_loader, args, env_model, wandb=None):        

        is_irm = self.irm_p > 0
        best_val_perf, best_epoch = 0, 0
        test_perf, best_test_perf = 0, 0

        for epoch in range(args.IL_epochs):
            self.model.train()
            loss, erm_loss, penalty = self.compute_il_loss(train_loader, env_model)

            print("Epoch {} | Train Loss = {:.4f} (ERM = {:.4f}, Penalty = {:.4f}) | Best Test perf = {:.4f}, ".format(epoch, loss, erm_loss, penalty, best_test_perf), flush=True)

            self.model.eval()
            # validation
            val_perf = self.test(valid_loader, args, env_model)
            test_perf = self.test(test_loader, args, env_model)
            if val_perf < best_val_perf:
                cnt += epoch >= args.pretrain
            else:
                cnt = (cnt + int(epoch >= args.pretrain)) if best_val_perf == 1.0 else 0
                best_val_perf = val_perf
                best_epoch = epoch
                best_test_perf = test_perf
        
            if epoch >= args.pretrain and cnt >= args.early_stopping:
                print("Early stopping at epoch {}.".format(epoch))
                if is_irm:
                    print("Invariant Learning is done.")
                else:
                    print("ERM is done.")
                break
        return best_test_perf,best_val_perf, best_epoch
    
    @torch.no_grad()
    def test(self, loader, args, env_model):
        perf = eval_model(self.model, self.device, loader, args.evaluator, env_model=env_model, eval_metric=args.eval_metric)
        return perf


class EM_EDNIL_Trainer_EI_hier:
    def __init__(self, num_classes, model, pre_optimizer, optimizer, temperature, device, num_task=1):

        if num_task == 1:
            num_classes = num_classes
        else:
            num_classes = num_task
        
        if num_classes == 1:
            self.erm_criterion = nn.MSELoss(reduction='none')
        elif num_classes == 2:
            self.erm_criterion = nn.BCEWithLogitsLoss(reduction='none')
        elif num_classes > 2:
            self.erm_criterion = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise NotImplementedError
        self.num_classes = num_classes
        self.model = model
        self.model.to(device)
        self.pre_optimizer = pre_optimizer
        self.optimizer = optimizer
        self.softmin_fn = lambda losses: (-losses / temperature).softmax(dim=-1)
        self.device = device
        
    def compute_erm_loss(self, preds, y, return_preds=False):
        assert len(preds.shape) == 2
        if self.num_classes > 2:
            ys = y.repeat(1, preds.shape[-1])  # (N, C)
            preds = preds.repeat(1, self.num_classes)  # (N, C * E)
            # preds = preds.reshape(preds.shape[0], self.num_classes, -1)  # (N, C, E)
        else:
            ys = y.reshape(-1, 1).repeat(1, preds.shape[-1])   # (N, E)

        assert ys.shape == (preds.shape[0], preds.shape[-1])
        # if self.num_classes == 2:
        ys = ys.float()
        losses = self.erm_criterion(preds, ys)  # (N, E)
        assert losses.shape == (preds.shape[0], preds.shape[-1])
        if return_preds:
            return losses, preds
        else:
            return losses
    
    def compute_ed_loss(self, env_probs, envw_thres=1):
        assert envw_thres >= 1
        env_log_probs = env_probs.clamp_min(1e-10).log()
        env_mask = F.one_hot(env_log_probs.argmax(-1), env_log_probs.shape[1])
        if envw_thres > 1:
            env_counts = env_mask.sum(0)
            env_w = env_counts.max() / env_counts.clamp(min=1e-10)
            env_mask = env_mask * env_w.clamp(max=envw_thres).reshape(1, -1)
        assert env_mask.shape == env_log_probs.shape
        ed_loss = -(env_mask * env_log_probs).sum() / env_mask.sum()
        return ed_loss
    
    def compute_contrast_loss(self, graph, labels, pre_labels=None, contrast_t=1.0, sampling='mul', y_pred=None):
        causal_rep = F.normalize(graph) # important otherwise loss->nan.
        if sampling.lower() in ['mul', 'var']: 
            # modified from https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11
            device = causal_rep.device
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(1).T).float().to(device)
            if pre_labels is not None:
                pre_mask = torch.eq(pre_labels.unsqueeze(1), pre_labels.unsqueeze(1).T).float().to(device)
                mask += pre_mask
                mask[mask>1] = 1
                
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
            # logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                        # torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
            # mask = mask * logits_mask
            # compute log_prob
            exp_logits = torch.exp(logits) #* logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            # print(log_prob)
            # print(mask.sum(1))
            # compute mean of log-likelihood over positive
            is_valid = mask.sum(1) != 0
            mean_log_prob_pos = (mask * log_prob).sum(1)[is_valid] / mask.sum(1)[is_valid]
            # some classes may not be sampled by more than 2
            mean_log_prob_pos[torch.isnan(mean_log_prob_pos)] = 0.0

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

    def train_EI_hier_consistent(self, train_loader, args, wandb=None):
        loss_curv, min_loss, best_ep, best_para = [], None, None, {}
        for ep in range(args.EI_epochs):            
            self.model.train()
            accum_loss, accum_ed_loss, accum_erm_loss = 0, 0, 0
            accum_causal_contrast_loss, accum_spu_contrast_loss = 0, 0
            for graph in train_loader:
                graph = graph.to(self.device)
                y = graph.y
                loss, ed_loss = 0, 0
                causal_contrast_loss, spu_contrast_loss = 0, 0
                pre_env = None
                edge_mask = torch.full((graph.edge_index.shape[1],), -1, device=graph.edge_index.device)
                for hier in range(self.model.num_hieararchy):
                    env_preds, preds, graph, c_rep, s_rep, edge_mask = self.model(graph, hier, edge_mask)   # (N, E) or (N, C * E)
                    erm_losses = self.compute_erm_loss(env_preds, y)
                    env_probs = self.softmin_fn(erm_losses)  # (N, E)
                    ed_loss += self.compute_ed_loss(env_probs, envw_thres=args.envw_thres)
                    causal_contrast_loss += self.compute_contrast_loss(c_rep, y) # ciga loss
                    env = env_probs.argmax(-1)                   
                    spu_contrast_loss += self.compute_contrast_loss(s_rep, env, pre_env) # me loss
                    loss += ed_loss + args.beta * causal_contrast_loss + args.alpha * spu_contrast_loss    
                    pre_env = env


                y_loss = self.compute_erm_loss(preds, y).mean()
                y_loss = torch.tensor(0.).to(self.device) 
                loss += y_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()

                accum_loss += loss.detach().item()
                accum_erm_loss += y_loss.detach().item()
                accum_ed_loss += ed_loss.detach().item()
                accum_causal_contrast_loss += causal_contrast_loss.detach().item()
                accum_spu_contrast_loss += spu_contrast_loss.detach().item()
                
            accum_loss /= len(train_loader.dataset)
            accum_erm_loss /= len(train_loader.dataset)
            accum_ed_loss /= len(train_loader.dataset)
            accum_causal_contrast_loss /= len(train_loader.dataset)
            accum_spu_contrast_loss /= len(train_loader.dataset)
            
            print("Epoch {} | Train Loss = {:.4f}".format(ep, accum_loss), flush=True)
            loss_curv.append(accum_loss)

            if best_ep is None or accum_loss < min_loss:
                min_loss = accum_loss
                best_ep = ep
        return best_ep, min_loss
