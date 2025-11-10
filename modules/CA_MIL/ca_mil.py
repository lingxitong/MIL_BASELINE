import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class CA_MIL(nn.Module):
    """
    Context-Aware Multiple Instance Learning for WSI Classification (ICLR 2024)
    https://arxiv.org/pdf/2305.05314
    
    The key idea is to incorporate contextual information between instances
    using attention mechanisms that consider relationships between patches.
    """
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512, rrt=None):
        super(CA_MIL, self).__init__()
        self.rrt = rrt
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        self.K = 1
        
        # Feature transformation
        self.feature = [nn.Linear(in_dim, self.L)]
        self.feature += [act]
        
        if dropout:
            self.feature += [nn.Dropout(dropout)]
            
        if self.rrt is not None:
            self.feature += [self.rrt]
        
        self.feature = nn.Sequential(*self.feature)
        
        # Context attention mechanism
        # Query, Key, Value projections for self-attention
        self.query = nn.Linear(self.L, self.D)
        self.key = nn.Linear(self.L, self.D)
        self.value = nn.Linear(self.L, self.D)
        
        # Instance-level attention (after context aggregation)
        self.attention = nn.Sequential(
            nn.Linear(self.D, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.D * self.K, self.num_classes),
        )
        
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        
        self.apply(initialize_weights)
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        
        # Feature transformation
        h = self.feature(x)  # [B, N, L] or [N, L]
        h = h.squeeze(0)  # [N, L]
        
        # Context-aware attention mechanism
        # Compute queries, keys, values
        Q = self.query(h)  # [N, D]
        K = self.key(h)    # [N, D]
        V = self.value(h)  # [N, D]
        
        # Scaled dot-product attention to capture context
        scale = self.D ** -0.5
        attn_scores = torch.matmul(Q, K.transpose(0, 1)) * scale  # [N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N, N]
        
        # Aggregate context information
        context_features = torch.matmul(attn_weights, V)  # [N, D]
        context_features = self.dropout(context_features)
        
        # Instance-level attention for bag aggregation
        A = self.attention(context_features)  # [N, K]
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # [K, N]
        A = F.softmax(A, dim=-1)  # softmax over N
        
        # Aggregate to bag level
        M = torch.mm(A, context_features)  # [K, D]
        
        # Classification
        logits = self.classifier(M)  # [K, num_classes]
        
        forward_return['logits'] = logits
        
        if return_WSI_feature:
            forward_return['WSI_feature'] = M
        
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori
        
        return forward_return
