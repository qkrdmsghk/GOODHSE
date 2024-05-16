# The model implementation is adopted from the dgllife library
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software in this distribution may have been modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.utils import degree
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from drugood.models import BACKBONES
from models.conv import GINConv, GCNConv

__all__ = ['GNN_node']

@BACKBONES.register_module()
### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """

    def __init__(self,
                 num_layer,
                 emb_dim,
                 input_dim=1,
                 drop_ratio=0.5,
                 JK="last",
                 residual=False,
                 gnn_type='gin',
                 edge_dim=-1):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers

        '''

        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        # if self.num_layer < 2:
        #     raise ValueError("Number of GNN layers must be greater than 1.")

        if input_dim == 9:
            # ogb dataset
            self.node_encoder = AtomEncoder(emb_dim)  # uniform input node embedding
            self.edge_dim = 1
        elif input_dim == -1:
            # ogbg-ppa
            self.node_encoder = torch.nn.Embedding(1, emb_dim)  # uniform input node embedding
            self.edge_dim = 7
        elif edge_dim != -1:
            # drugood
            self.node_encoder = torch.nn.Linear(input_dim, emb_dim)  # uniform input node embedding
            self.edge_dim = edge_dim
        else:
            # only for spmotif dataset
            self.node_encoder = torch.nn.Linear(input_dim, emb_dim)
            self.edge_dim = -1
        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GINConv(emb_dim, edge_dim=self.edge_dim))
            elif gnn_type == 'gcn':
                self.convs.append(GCNConv(emb_dim, edge_dim=self.edge_dim))
            else:
                ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        ### computing input node embedding
        h_list = [self.node_encoder(x)]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer):
                node_representation += h_list[layer]

        return node_representation