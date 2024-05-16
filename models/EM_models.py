import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data.batch as DataBatch
from torch_geometric.nn import (ASAPooling, global_add_pool, global_max_pool,
                                global_mean_pool)
from utils.get_subgraph import relabel, split_batch
from utils.mask import clear_masks, set_masks
from models.gnn_EM import GNN_node
from torch_geometric.utils import softmax, add_remaining_self_loops



class MainModel(nn.Module):
    def __init__(self, args, input_dim, num_class, num_task=1, JK='last', residual=False, gnn_type="gin"):
        super(MainModel, self).__init__()
        self.gnn_encoder = GNN_node(num_layer=args.IL_num_layers,
                                    emb_dim=args.emb_dim,
                                    input_dim=input_dim,
                                    drop_ratio=args.dropout,
                                    JK=JK,
                                    residual=residual,
                                    gnn_type=gnn_type,
                                    edge_dim=args.edge_dim)
        self.graph_pooling = args.pooling
        self.il_cls = args.il_cls
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        if num_task > 1:
            num_class = num_task

        if 'linear' in self.il_cls:
            self.graph_pred = torch.nn.Linear(args.emb_dim, num_class)
        elif 'mlp' in self.il_cls:
            self.graph_pred = torch.nn.Sequential(torch.nn.Linear(args.emb_dim, args.emb_dim),
                                                    torch.nn.ReLU(),
                                                    torch.nn.Linear(args.emb_dim, num_class))
        
    def forward(self, batch, env_model=None):
        h_node = self.gnn_encoder(batch)
        h_graph = self.pool(h_node, batch.batch)
        if env_model != None and 'env_model' in self.il_cls:
            for i in range(env_model.num_hieararchy):
                _, _, batch, c_rep, s_rep = env_model(batch, i)
            h_graph += c_rep
        pred = self.graph_pred(h_graph)
        return pred, h_graph

class DecompModel(nn.Module):
    def __init__(self, args, input_dim=39, JK="last", residual=False, gnn_type="gin"):
        super(DecompModel, self).__init__()
        self.gnn_encoder = GNN_node(num_layer=args.EI_num_layers,
                                    emb_dim=args.emb_dim,
                                    input_dim=input_dim,
                                    drop_ratio=args.dropout,
                                    JK=JK,
                                    residual=residual,
                                    gnn_type=gnn_type,
                                    edge_dim=args.edge_dim)
        self.ratio = args.r
        self.temperature = args.temperature
        self.edge_att = nn.Sequential(nn.Linear(args.emb_dim * 2, args.emb_dim * 4), nn.ReLU(), nn.Linear(args.emb_dim * 4, 1))
        
     
    def split_graph(self, data, edge_score, ratio):
        # Adopt from GOOD benchmark to improve the efficiency
        from torch_geometric.utils import degree
        def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
            r'''
            Adopt from <https://github.com/rusty1s/pytorch_scatter/issues/48>_.
            '''
            f_src = src.float()
            f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
            norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
            perm = norm.argsort(dim=dim, descending=descending)

            return src[perm], perm

        def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
            rank, perm = sparse_sort(src, index, dim, descending, eps)
            num_nodes = degree(index, dtype=torch.long)
            k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
            start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
            mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
            mask = torch.cat(mask, dim=0)
            mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
            topk_perm = perm[mask]
            exc_perm = perm[~mask]

            return topk_perm, exc_perm, rank, perm, mask

        has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None
        new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio, descending=True)
        new_causal_edge_index = data.edge_index[:, new_idx_reserve]
        new_spu_edge_index = data.edge_index[:, new_idx_drop]

        new_causal_edge_weight = edge_score[new_idx_reserve]
        new_spu_edge_weight = -edge_score[new_idx_drop]

        if has_edge_attr:
            new_causal_edge_attr = data.edge_attr[new_idx_reserve]
            new_spu_edge_attr = data.edge_attr[new_idx_drop]
        else:
            new_causal_edge_attr = None
            new_spu_edge_attr = None

        return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
            (new_spu_edge_index, new_spu_edge_attr, new_spu_edge_weight)
            
    
    def stochastic_split_graph(self, data, edge_score, edge_mask, hierarchy=None):
        
        def sparse_stochastic(edge_score, edge_mask, index, temperature, threshold, ):
            weights = softmax(edge_score, index)
            sample_prob = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(temperature, probs=weights)
            y = sample_prob.rsample()
            y_hard = (y>threshold).to(y.dtype)
            y = (y_hard - y).detach() + y
            edge_mask[y==1] = hierarchy+1
            c_idx = edge_mask > 0
            s_idx = edge_mask == -1
            return c_idx, s_idx, edge_mask

        has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None
        
        new_idx_reserve, new_idx_drop, edge_mask = sparse_stochastic(edge_score, edge_mask, data.edge_index[0], self.temperature, self.ratio)
        # new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio, descending=True)
        new_causal_edge_index = data.edge_index[:, new_idx_reserve]
        new_spu_edge_index = data.edge_index[:, new_idx_drop]
        new_causal_edge_weight = edge_score[new_idx_reserve]
        new_spu_edge_weight = -edge_score[new_idx_drop]

        if has_edge_attr:
            new_causal_edge_attr = data.edge_attr[new_idx_reserve]
            new_spu_edge_attr = data.edge_attr[new_idx_drop]
        else:
            new_causal_edge_attr = None
            new_spu_edge_attr = None

        return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
            (new_spu_edge_index, new_spu_edge_attr, new_spu_edge_weight), edge_mask


    def forward(self, batch, edge_mask, return_data="rep", debug=False, hierarchy=None):
        # obtain the graph embeddings from the featurizer GNN encoder
        h = self.gnn_encoder(batch)
        device = h.device
        # seperate the input graphs into \hat{G_c} and \hat{G_s}
        # using edge-level attetion
        row, col = batch.edge_index
        if batch.edge_attr == None:
            batch.edge_attr = torch.ones(row.size(0)).to(device)
        edge_rep = torch.cat([h[row], h[col]], dim=-1)
        pred_edge_weight = self.edge_att(edge_rep).view(-1)

        (causal_edge_index, causal_edge_attr, causal_edge_weight), \
            (spu_edge_index, spu_edge_attr, spu_edge_weight), edge_mask = self.stochastic_split_graph(batch, pred_edge_weight, edge_mask, hierarchy=hierarchy)
        if return_data == "raw":
            causal_x, causal_edge_index, causal_batch, _ = relabel(batch.x, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(batch.x, spu_edge_index, batch.batch)
        elif return_data == "rep":
            causal_x, causal_edge_index, causal_batch, _ = relabel(h, causal_edge_index, batch.batch)
            spu_x, spu_edge_index, spu_batch, _ = relabel(h, spu_edge_index, batch.batch)
        else:
            raise Exception("Not implemented return data type")
        # obtain \hat{G_c}
        
        causal_graph = DataBatch.Batch(batch=causal_batch,
                                       edge_index=causal_edge_index,
                                       x=causal_x,
                                       edge_attr=causal_edge_attr,
                                       y=batch.y,
                                    #    group=batch.group,
                                    #    idx=batch.idx,
                                       ptr=batch.ptr)
        spu_graph = DataBatch.Batch(batch=spu_batch,
                                        edge_index=spu_edge_index,
                                        x=spu_x,
                                        edge_attr=spu_edge_attr,
                                        y=batch.y,
                                        # group=batch.group,
                                        # idx=batch.idx,
                                        ptr=batch.ptr)
        
        graph = DataBatch.Batch(batch=batch.batch,
                                edge_index=batch.edge_index,
                                x=h,
                                edge_attr=batch.edge_attr,
                                y=batch.y,
                                # group=batch.group,
                                # idx=batch.idx,
                                ptr=batch.ptr)

        return causal_graph, spu_graph, graph, edge_mask


class DomainHierarchyClassifier(torch.nn.Module):
    def __init__(self, args, num_task, num_domain, num_class):
        super(DomainHierarchyClassifier, self).__init__()
        self.num_hieararchy = len(num_domain)
        self.num_task = num_task
        self.decomp_models = []
        self.env_predictors = []
        self.label_predictors = []
        self.s_embs = []
        self.c_embs = []
        if num_task > 1:
            num_class = num_task
        for i in range(self.num_hieararchy):
            if i !=0:
                self.decomp_models.append(DecompModel(args, input_dim=args.emb_dim))
            else:
                self.decomp_models.append(DecompModel(args, input_dim=args.input_dim))
            
            self.s_embs.append(torch.nn.Sequential(torch.nn.Linear(args.emb_dim, args.emb_dim), nn.ReLU(), nn.Linear(args.emb_dim, args.emb_dim)))
            self.c_embs.append(torch.nn.Sequential(torch.nn.Linear(args.emb_dim, args.emb_dim), nn.ReLU(), nn.Linear(args.emb_dim, args.emb_dim)))
            self.env_predictors.append(torch.nn.Linear(args.emb_dim + num_task, num_domain[i]))
            self.label_predictors.append(torch.nn.Linear(args.emb_dim, num_class))
        
        self.decomp_models = torch.nn.ModuleList(self.decomp_models)
        self.s_embs = torch.nn.ModuleList(self.s_embs)
        self.c_embs = torch.nn.ModuleList(self.c_embs)
        self.env_predictors = torch.nn.ModuleList(self.env_predictors)
        self.label_predictors = torch.nn.ModuleList(self.label_predictors)
        

    def forward(self, batched_data, hierarchy, edge_mask, debug=False):
        c_graph, s_graph, batched_data, edge_mask = self.decomp_models[hierarchy](batched_data, edge_mask, return_data="rep", hierarchy=hierarchy)
        # print('hier {} - c_graph rate {:.4f}, s_graph rate {:.4f}'.format(hierarchy, c_graph.edge_index.shape[1]/batched_data.edge_index.shape[1], s_graph.edge_index.shape[1]/batched_data.edge_index.shape[1]))
        
        c_rep = global_add_pool(c_graph.x, c_graph.batch)
        s_rep = global_add_pool(s_graph.x, s_graph.batch)
        
        c_rep = self.c_embs[hierarchy](c_rep)
        s_rep = self.s_embs[hierarchy](s_rep)
        
        # bug..! 24/01/25
        if  c_rep.shape[0] != s_rep.shape[0]:
            s_rep = torch.cat([s_rep, torch.zeros(c_rep.shape[0]-s_rep.shape[0], s_rep.shape[1]).to(s_rep.device)], dim=0)
            # print(s_rep.shape)
        
        assert c_rep.shape == s_rep.shape

        y_part = torch.nan_to_num(batched_data.y).float()
        y_part = y_part.reshape(len(y_part), self.num_task)
        
        env_preds = self.env_predictors[hierarchy](torch.cat([s_rep, y_part], dim=1))
        preds = self.label_predictors[hierarchy](c_rep)
        if debug:
            return env_preds, preds, batched_data, c_rep, s_rep, edge_mask, c_graph, s_graph
        else:
            return env_preds, preds, batched_data, c_rep, s_rep, edge_mask

# class DomainClassifier(torch.nn.Module):
#     def __init__(self, backend_dim, decomp_model, backend, num_domain, num_task):
#         super(DomainClassifier, self).__init__()
#         self.backend = backend
#         self.num_task = num_task
#         if decomp_model is not None:
#             self.decomp_model = decomp_model
#         else:
#             self.decomp_model = None
#         self.predictor = torch.nn.Linear(backend_dim + num_task, num_domain)

#     def forward(self, batched_data):
#         c_graph, s_graph, batched_data = self.decomp_model(batched_data, return_data="raw")
#         graph_pred, graph_feat = self.backend(s_graph, get_rep=True)
#         y_part = torch.nan_to_num(batched_data.y).float()
#         y_part = y_part.reshape(len(y_part), self.num_task)
#         return self.predictor(torch.cat([graph_feat, y_part], dim=-1)), batched_data

#     def _get_split_graph(self, batched_data):
#         if self.decomp_model is not None:
#             c_graph, s_graph, _ = self.decomp_model(batched_data, return_data="raw")
#         return c_graph, s_graph
    
    