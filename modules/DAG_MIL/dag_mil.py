import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentionalAggregation


class DAG_MIL(nn.Module):
    def __init__(self, dim_in, dim_hidden, n_classes, topk, stride, agg_type='bi-interaction', dropout=0.3):
        super().__init__()
        self.topk = topk
        self.stride = stride
        self.scale = dim_hidden ** -0.5
        self.agg_type = agg_type

        self._fc1 = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.LeakyReLU())
        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.offset_net = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, topk * 2),
            nn.Sigmoid()
        )

        if self.agg_type == 'gcn':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = None
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError

        self.out_linear = nn.Linear(dim_hidden, dim_hidden)
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, n_classes)

        self.att_net = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden // 2, 1)
        )
        self.readout = AttentionalAggregation(self.att_net)
        self.expand_alpha = nn.Parameter(torch.randn(1))

    def forward(self, x, coords, return_WSI_feature=False):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        x = x.float()
        coords = coords.float()

        B, N, _ = x.shape
        x = self._fc1(x)
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5

        e_h = self.W_head(x)
        e_t = self.W_tail(x)

        offset = self.offset_net(e_h).reshape(B, N, self.topk, 2)
        offset = offset * self.stride * math.sqrt(max(N, 1)) * torch.sigmoid(self.expand_alpha)
        query_coords = coords.unsqueeze(2) + offset
        dist = torch.cdist(query_coords.view(B, N * self.topk, 2), coords, p=2)
        knn_index = dist.argmin(dim=-1).view(B, N, self.topk)

        batch_indices = torch.arange(B, device=knn_index.device).view(-1, 1, 1)
        nb_h = e_t[batch_indices, knn_index, :]

        e_h_norm = F.normalize(e_h, dim=-1)
        nb_h_norm = F.normalize(nb_h, dim=-1)
        h_expand = e_h_norm.unsqueeze(2).expand(-1, -1, self.topk, -1)
        sim_score = torch.sum(h_expand * nb_h_norm, dim=-1)
        edge_weight = F.softmax(sim_score, dim=-1)

        eh_r = edge_weight.unsqueeze(-1) * nb_h
        gate = torch.tanh(h_expand + eh_r)
        ka_weight = torch.einsum("bnkd,bnkd->bnk", nb_h, gate)
        ka_prob = F.softmax(ka_weight, dim=-1).unsqueeze(2)
        e_nh = torch.matmul(ka_prob, nb_h).squeeze(2)

        if self.agg_type == 'gcn':
            embedding = e_h + e_nh
            embedding = self.activation(self.linear1(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_nh))
            bi_embedding = self.activation(self.linear2(e_h * e_nh))
            embedding = sum_embedding + bi_embedding
        else:
            raise NotImplementedError

        embedding = self.activation(self.out_linear(embedding))
        h = self.message_dropout(embedding)

        pooled_list = []
        for b in range(B):
            pooled_list.append(self.readout(h[b]))
        wsi_feature = torch.stack(pooled_list, dim=0)
        wsi_feature = self.norm(wsi_feature)
        logits = self.fc(wsi_feature)

        forward_return = {'logits': logits}
        if return_WSI_feature:
            forward_return['WSI_feature'] = wsi_feature
        return forward_return
