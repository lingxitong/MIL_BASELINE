"""
PSEBMIX_MIL: Pseudo-bag Mixup for MIL
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


def psebmix_data(features1, label1, features2, label2, n_pseudo_bags=30, alpha=1.0):
    """Pseudo-bag Mixup (PseMix)"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    device = features1.device
    
    # Simplified pseudo-bag division: based on random assignment
    def simple_pseudo_bag_divide(features, n_bags):
        n_instances = features.size(0)
        if n_instances < n_bags:
            n_bags = n_instances
        bag_assignments = torch.randint(0, n_bags, (n_instances,), device=features.device)
        return bag_assignments
    
    # Divide into pseudo-bags
    pseb_ind1 = simple_pseudo_bag_divide(features1, n_pseudo_bags)
    pseb_ind2 = simple_pseudo_bag_divide(features2, n_pseudo_bags)
    
    # Select number of pseudo-bags to mix
    n_select = int(n_pseudo_bags * lam)
    n_select = max(1, min(n_select, n_pseudo_bags - 1))
    
    # Select pseudo-bags from both bags
    selected_psebs1 = torch.randperm(n_pseudo_bags)[:n_select]
    selected_psebs2 = torch.randperm(n_pseudo_bags)[:(n_pseudo_bags - n_select)]
    
    # Combine selected pseudo-bags
    mask1 = torch.zeros(features1.size(0), dtype=torch.bool, device=features1.device)
    for pseb_id in selected_psebs1:
        mask1 |= (pseb_ind1 == pseb_id)
    
    mask2 = torch.zeros(features2.size(0), dtype=torch.bool, device=features2.device)
    for pseb_id in selected_psebs2:
        mask2 |= (pseb_ind2 == pseb_id)
    
    selected1 = features1[mask1] if mask1.sum() > 0 else torch.empty(0, features1.size(1), device=features1.device)
    selected2 = features2[mask2] if mask2.sum() > 0 else torch.empty(0, features2.size(1), device=features2.device)
    
    if selected1.size(0) > 0 and selected2.size(0) > 0:
        mixed_features = torch.cat([selected1, selected2], dim=0)
        content_ratio = n_select / n_pseudo_bags
    elif selected1.size(0) > 0:
        mixed_features = selected1
        content_ratio = 1.0
    else:
        mixed_features = selected2
        content_ratio = 0.0
    
    mixed_label = content_ratio * label1 + (1 - content_ratio) * label2
    
    return mixed_features, mixed_label, content_ratio


class PSEBMIX_MIL(nn.Module):
    """AB_MIL with Pseudo-bag Mixup augmentation"""
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512):
        super(PSEBMIX_MIL, self).__init__()
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

