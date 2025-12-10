"""
INSMIX_MIL: Instance-level Mixup for MIL
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


def insmix_data(features1, label1, features2, label2, alpha=1.0):
    """Instance-level Mixup (InsMix)"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    n1, n2 = features1.size(0), features2.size(0)
    
    # Randomly select instances for mixing
    num_select1 = int(n1 * lam)
    num_select2 = n2 - num_select1
    
    if num_select1 > 0 and num_select1 < n1:
        indices1 = torch.randperm(n1)[:num_select1]
        selected1 = features1[indices1]
    elif num_select1 >= n1:
        selected1 = features1
    else:
        selected1 = torch.empty(0, features1.size(1), device=features1.device)
    
    if num_select2 > 0 and num_select2 < n2:
        indices2 = torch.randperm(n2)[:num_select2]
        selected2 = features2[indices2]
    elif num_select2 >= n2:
        selected2 = features2
    else:
        selected2 = torch.empty(0, features2.size(1), device=features2.device)
    
    if selected1.size(0) > 0 and selected2.size(0) > 0:
        mixed_features = torch.cat([selected1, selected2], dim=0)
        area_ratio = selected1.size(0) / mixed_features.size(0)
    elif selected1.size(0) > 0:
        mixed_features = selected1
        area_ratio = 1.0
    else:
        mixed_features = selected2
        area_ratio = 0.0
    
    mixed_label = area_ratio * label1 + (1 - area_ratio) * label2
    
    return mixed_features, mixed_label, area_ratio


class INSMIX_MIL(nn.Module):
    """AB_MIL with Instance-level Mixup augmentation"""
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512):
        super(INSMIX_MIL, self).__init__()
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

