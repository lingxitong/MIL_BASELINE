"""
RANKMIX_MIL: Attention-based Rank Mixup for MIL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def rankmix_data(features1, label1, features2, label2, model, alpha=1.0, strategy='rank'):
    """RankMix: attention-based mixing"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    device = features1.device
    
    # Get attention scores
    with torch.no_grad():
        out1 = model(features1.unsqueeze(0), return_WSI_attn=True)
        out2 = model(features2.unsqueeze(0), return_WSI_attn=True)
        attn1 = out1['WSI_attn'].squeeze() if 'WSI_attn' in out1 else torch.ones(features1.size(0), device=device)
        attn2 = out2['WSI_attn'].squeeze() if 'WSI_attn' in out2 else torch.ones(features2.size(0), device=device)
    
    n1, n2 = features1.size(0), features2.size(0)
    
    if strategy == 'rank':
        # Selection based on attention ranking
        _, sorted_idx1 = torch.sort(attn1, descending=True)
        _, sorted_idx2 = torch.sort(attn2, descending=True)
        
        num_select1 = int(n1 * lam)
        num_select2 = int(n2 * (1 - lam))
        
        selected1 = features1[sorted_idx1[:num_select1]] if num_select1 > 0 else torch.empty(0, features1.size(1), device=device)
        selected2 = features2[sorted_idx2[:num_select2]] if num_select2 > 0 else torch.empty(0, features2.size(1), device=device)
    else:  # 'shrink' strategy
        min_n = min(n1, n2)
        
        if n1 > n2:
            _, sorted_idx = torch.sort(attn1, descending=True)
            selected1 = features1[sorted_idx[:min_n]]
            selected2 = features2
        else:
            _, sorted_idx = torch.sort(attn2, descending=True)
            selected1 = features1
            selected2 = features2[sorted_idx[:min_n]]
        
        selected1 = lam * selected1
        selected2 = (1 - lam) * selected2
    
    if selected1.size(0) > 0 and selected2.size(0) > 0:
        mixed_features = torch.cat([selected1, selected2], dim=0)
        mix_ratio = selected1.size(0) / (selected1.size(0) + selected2.size(0))
    elif selected1.size(0) > 0:
        mixed_features = selected1
        mix_ratio = 1.0
    else:
        mixed_features = selected2
        mix_ratio = 0.0
    
    mixed_label = mix_ratio * label1 + (1 - mix_ratio) * label2
    
    return mixed_features, mixed_label, mix_ratio


class RANKMIX_MIL(nn.Module):
    """AB_MIL with RankMix augmentation"""
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512):
        super(RANKMIX_MIL, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        self.K = 1
        self.feature = [nn.Linear(in_dim, self.L)]
        self.feature += [act]
        if dropout:
            self.feature += [nn.Dropout(dropout)]
        self.feature = nn.Sequential(*self.feature)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes),
        )
        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        feature = self.feature(x)
        feature = feature.squeeze(0)
        A = self.attention(feature)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)
        A = F.softmax(A, dim=-1)
        M = torch.mm(A, feature)
        logits = self.classifier(M)
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = M
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori
        return forward_return

