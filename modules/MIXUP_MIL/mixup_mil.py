"""
MIXUP_MIL: Standard Mixup for MIL
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


def mixup_data(features1, label1, features2, label2, alpha=1.0):
    """Standard Mixup method"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    # Handle bags of different sizes
    n1, n2 = features1.size(0), features2.size(0)
    max_n = max(n1, n2)
    
    # Resample to the same size
    if n1 < max_n:
        indices = torch.randint(0, n1, (max_n,))
        features1 = features1[indices]
    if n2 < max_n:
        indices = torch.randint(0, n2, (max_n,))
        features2 = features2[indices]
    
    mixed_features = lam * features1 + (1 - lam) * features2
    mixed_label = lam * label1 + (1 - lam) * label2
    
    return mixed_features, mixed_label, lam


class MIXUP_MIL(nn.Module):
    """AB_MIL with Standard Mixup augmentation"""
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512):
        super(MIXUP_MIL, self).__init__()
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

