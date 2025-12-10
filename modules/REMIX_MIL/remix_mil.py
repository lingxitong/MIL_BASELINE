"""
REMIX_MIL: Feature-level ReMix for MIL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def remix_data(features1, label1, features2, label2, mode='replace', rate=0.3, strength=0.5):
    """ReMix: feature-level mixing"""
    features1_np = features1.cpu().numpy()
    features2_np = features2.cpu().numpy()
    
    mixed_features = [f.copy() for f in features1_np]
    
    # Find nearest neighbors
    if features1_np.shape[0] > 0 and features2_np.shape[0] > 0:
        closest_idxs = np.argmin(cdist(features1_np, features2_np), axis=1)
        
        for i in range(len(features1_np)):
            if np.random.rand() <= rate:
                if mode == 'replace':
                    mixed_features[i] = features2_np[closest_idxs[i]]
                elif mode == 'append':
                    mixed_features.append(features2_np[closest_idxs[i]])
                elif mode == 'interpolate':
                    generated = (1 - strength) * mixed_features[i] + strength * features2_np[closest_idxs[i]]
                    mixed_features.append(generated)
    
    mixed_features = torch.from_numpy(np.array(mixed_features)).to(features1.device).float()
    
    # Label mixing (simplified to uniform mixing)
    mixed_label = 0.5 * label1 + 0.5 * label2
    
    return mixed_features, mixed_label, 0.5


class REMIX_MIL(nn.Module):
    """AB_MIL with ReMix augmentation"""
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512):
        super(REMIX_MIL, self).__init__()
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

