"""
RRT_MIL - Feature Re-Embedding Multiple Instance Learning
Reference: https://github.com/DearCaat/RRT-MIL
Paper: Feature Re-Embedding: Towards Foundation Model-Level Performance in Computational Pathology (CVPR 2024)
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

class RRTReEmbedding(nn.Module):
    """
    RRT Re-Embedding Module
    Re-embeds features to enhance representation learning
    """
    def __init__(self, in_dim, out_dim=None):
        super(RRTReEmbedding, self).__init__()
        self.out_dim = out_dim if out_dim is not None else in_dim
        self.re_embed = nn.Sequential(
            nn.Linear(in_dim, self.out_dim),
            nn.LayerNorm(self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim)
        )
    
    def forward(self, x):
        return self.re_embed(x) + x  # Residual connection

class RRT_MIL(nn.Module):
    """
    Feature Re-Embedding Multiple Instance Learning
    
    This model uses feature re-embedding mechanism before bag-level aggregation:
    - Re-embeds patch features to enhance representation
    - Uses attention mechanism for aggregation (similar to ABMIL)
    
    Based on ABMIL architecture with RRT re-embedding module.
    Reference: https://github.com/DearCaat/RRT-MIL
    """
    def __init__(self, L=512, D=128, num_classes=2, dropout=0.1, act=nn.ReLU(), in_dim=512):
        super(RRT_MIL, self).__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        self.K = 1
        
        # Feature encoder (same as ABMIL)
        self.feature = nn.Sequential(
            nn.Linear(in_dim, self.L),
            act
        )
        
        if dropout:
            self.feature.add_module('dropout', nn.Dropout(dropout))
        
        # RRT Re-embedding module
        self.rrt_reembed = RRTReEmbedding(self.L, self.L)
        
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
        
        # RRT Re-embedding
        feature = self.rrt_reembed(feature)  # (N, L)
        
        # Attention aggregation
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

