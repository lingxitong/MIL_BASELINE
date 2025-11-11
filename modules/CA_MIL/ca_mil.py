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

class ContextAwareAttention(nn.Module):
    """
    Context-aware attention mechanism that considers relationships between instances
    """
    def __init__(self, L, D):
        super(ContextAwareAttention, self).__init__()
        self.L = L
        self.D = D
        
        # Query, Key, Value projections
        self.query = nn.Linear(L, D)
        self.key = nn.Linear(L, D)
        self.value = nn.Linear(L, D)
        
        # Attention weights
        self.attention_weights = nn.Sequential(
            nn.Linear(D, D),
            nn.Tanh(),
            nn.Linear(D, 1)
        )
        
    def forward(self, x):
        """
        x: [N, L] where N is number of instances
        """
        Q = self.query(x)  # [N, D]
        K = self.key(x)    # [N, D]
        V = self.value(x)  # [N, D]
        
        # Compute attention scores with context
        # Use scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.D ** 0.5)  # [N, N]
        context_weights = F.softmax(scores, dim=-1)  # [N, N]
        
        # Apply context to values
        context_features = torch.matmul(context_weights, V)  # [N, D]
        
        # Compute instance attention weights
        A = self.attention_weights(context_features)  # [N, 1]
        A = torch.transpose(A, -1, -2)  # [1, N]
        A = F.softmax(A, dim=-1)
        
        return A, context_features

class CA_MIL(nn.Module):
    """
    Context-Aware Multiple Instance Learning for WSI Classification (ICLR 2024)
    Key idea: Incorporates spatial and contextual relationships between patches
    using self-attention mechanisms.
    """
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512, rrt=None):
        super(CA_MIL, self).__init__()
        self.rrt = rrt
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        
        # Feature extraction layers
        self.feature = [nn.Linear(in_dim, self.L)]
        self.feature += [act]
        
        if dropout:
            self.feature += [nn.Dropout(dropout)]
            
        if self.rrt is not None:
            self.feature += [self.rrt]
        
        self.feature = nn.Sequential(*self.feature)
        
        # Context-aware attention mechanism
        self.context_attention = ContextAwareAttention(self.L, self.D)
        
        # Feature aggregation
        self.feature_aggregation = nn.Sequential(
            nn.Linear(self.D, self.L),
            act
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.L, self.num_classes)
        )
        
        self.apply(initialize_weights)
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        
        # Extract features
        feature = self.feature(x)
        feature = feature.squeeze(0)  # [N, L]
        
        # Apply context-aware attention
        A, context_features = self.context_attention(feature)  # A: [1, N], context_features: [N, D]
        
        # Aggregate context features
        aggregated_context = torch.mm(A, context_features)  # [1, D]
        
        # Transform back to feature space
        bag_feature = self.feature_aggregation(aggregated_context)  # [1, L]
        
        # Classification
        logits = self.classifier(bag_feature)  # [1, num_classes]
        
        forward_return['logits'] = logits
        
        if return_WSI_feature:
            forward_return['WSI_feature'] = bag_feature
        
        if return_WSI_attn:
            forward_return['WSI_attn'] = A.transpose(-1, -2)
        
        return forward_return
