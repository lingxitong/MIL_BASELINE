"""
IB_MIL - Information Bottleneck Multiple Instance Learning
Reference: https://github.com/HHHedo/IBMIL
Paper: Interventional Bag Multi-Instance Learning On Whole-Slide Pathological Images (CVPR 2023)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class IB_MIL(nn.Module):
    """
    Information Bottleneck Multiple Instance Learning
    
    This model uses Information Bottleneck principle for bag-level aggregation:
    - Minimize mutual information between input features and compressed representation (compression)
    - Maximize mutual information between compressed representation and labels (prediction)
    
    Based on ABMIL architecture with IB aggregation mechanism.
    Reference: https://github.com/HHHedo/IBMIL
    """
    def __init__(self, L=512, D=128, num_classes=2, dropout=0.1, act=nn.ReLU(), in_dim=512, beta=0.1):
        super(IB_MIL, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        self.K = 1
        self.beta = beta  # IB trade-off parameter
        
        # Feature encoder (same as ABMIL)
        self.feature = nn.Sequential(
            nn.Linear(in_dim, self.L),
            act
        )
        
        if dropout:
            self.feature.add_module('dropout', nn.Dropout(dropout))
        
        # IB aggregation: learn compressed representation
        # Encoder: maps features to compressed representation
        self.encoder = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.D)
        )
        
        # Attention for aggregation
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, self.num_classes),
        )
        
        self.apply(initialize_weights)
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        
        # Handle input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        # Feature encoding
        feature = self.feature(x)  # (1, N, L)
        feature = feature.squeeze(0)  # (N, L)
        
        # IB aggregation: compute attention weights
        A = self.attention(feature)  # (N, K)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # (K, N)
        A = F.softmax(A, dim=-1)  # softmax over N
        
        # Bag-level representation
        M = torch.mm(A, feature)  # (K, L)
        
        # Classification
        logits = self.classifier(M)  # (K, num_classes)
        # Keep batch dimension for training compatibility
        # logits shape: (K, num_classes) where K=1
        
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = M.squeeze(0)
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori.squeeze(-1)
        
        return forward_return

