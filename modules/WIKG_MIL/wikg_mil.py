import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGPooling, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.nn.aggr import AttentionalAggregation
class WIKG_MIL(nn.Module):
    def __init__(self,in_dim=512, act = nn.LeakyReLU(),dim_hidden=512, topk=6, num_classes=2, agg_type='bi-interaction', dropout=0.1, pool='attn'):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.agg_type = agg_type
        self.dropout = dropout
        self._fc1 = nn.Sequential(nn.Linear(in_dim, dim_hidden), act)
        
        self.W_head = nn.Linear(dim_hidden, dim_hidden)
        self.W_tail = nn.Linear(dim_hidden, dim_hidden)

        self.scale = dim_hidden ** -0.5
        self.topk = topk
        self.agg_type = agg_type

        self.gate_U = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_V = nn.Linear(dim_hidden, dim_hidden // 2)
        self.gate_W = nn.Linear(dim_hidden // 2, dim_hidden)

        if self.agg_type == 'gcn':
            self.linear = nn.Linear(dim_hidden, dim_hidden)
        elif self.agg_type == 'sage':
            self.linear = nn.Linear(dim_hidden * 2, dim_hidden)  
        elif self.agg_type == 'bi-interaction':
            self.linear1 = nn.Linear(dim_hidden, dim_hidden)
            self.linear2 = nn.Linear(dim_hidden, dim_hidden)
        else:
            raise NotImplementedError
        
        self.activation = act
        if dropout:
            self.message_dropout = nn.Dropout(self.dropout)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden))
        
        # self.pooling = SAGPooling(in_channels=dim_hidden, ratio=0.5)
        self.norm = nn.LayerNorm(dim_hidden)
        self.fc = nn.Linear(dim_hidden, self.num_classes)
        if pool == "attn":
            self.att_net=nn.Sequential(nn.Linear(dim_hidden, dim_hidden // 2), nn.LeakyReLU(), nn.Linear(dim_hidden//2, 1))     
            self.readout = GlobalAttention(self.att_net)


    def forward(self, x, return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        x = self._fc1(x)    # [B,N,C]

        # B, N, C = x.shape
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5  

        e_h = self.W_head(x)
        # e_r = self.W_edge(x)
        e_t = self.W_tail(x)

        # construct neighbour
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 1
        # attn_logit = (e_h) @ e_t.transpose(-2, -1)  # 2
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # btmk_weight, btm_index = torch.topk(attn_logit, k=self.topk, dim=-1, largest=False)


        topk_index = topk_index.to(torch.long)
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # shape: [1, 10000, 4]

        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # shape: [1, 1, 1]
        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # shape: [1, 10000, 4, 512]

        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))  # 1 pixel wise   2 matmul

        # gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        if self.agg_type == 'gcn':
            embedding = e_h + e_Nh
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'sage':
            embedding = torch.cat([e_h, e_Nh], dim=2)
            embedding = self.activation(self.linear(embedding))
        elif self.agg_type == 'bi-interaction':
            sum_embedding = self.activation(self.linear1(e_h + e_Nh))
            bi_embedding = self.activation(self.linear2(e_h * e_Nh))
            # cat_embedding = self.activation(self.linear3(torch.cat([e_h, e_Nh], dim=2)))
            embedding = sum_embedding + bi_embedding
            # embedding = bi_embedding + cat_embedding
        
        h = self.message_dropout(embedding)
        # h = (h + h.mean(dim=1, keepdim=True)) * 0.5

        # h = self.pooling(h.squeeze(0), edge_index)[0]
        h_ori = h.clone()
        h = self.readout(h.squeeze(0), batch=None)
        h = self.norm(h)
        logits = self.fc(h)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = h
        if return_WSI_attn:
            WSI_attn = self.att_net(h_ori).squeeze()
            forward_return['WSI_attn'] = WSI_attn
        return forward_return

            
