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

class ADD_MIL(nn.Module):
    """
    Additive MIL: Intrinsically Interpretable MIL for Pathology (NeurIPS 2022)
    https://arxiv.org/pdf/2206.01794
    
    The key idea is that each instance makes an additive contribution to the bag-level prediction.
    The model consists of:
    1. Feature transformation layer
    2. Instance-level scoring (predicting per-instance contributions)
    3. Additive aggregation (sum of instance contributions)
    """
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512, rrt=None):
        super(ADD_MIL, self).__init__()
        self.rrt = rrt
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.L = L
        self.D = D
        
        # Feature transformation
        self.feature = [nn.Linear(in_dim, self.L)]
        self.feature += [act]
        
        if dropout:
            self.feature += [nn.Dropout(dropout)]
            
        if self.rrt is not None:
            self.feature += [self.rrt]
        
        self.feature = nn.Sequential(*self.feature)
        
        # Instance-level scoring network
        # Each instance gets a score for each class
        self.scoring = nn.Sequential(
            nn.Linear(self.L, self.D),
            act,
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(self.D, self.num_classes)
        )
        
        # Bias term for bag-level prediction
        self.bias = nn.Parameter(torch.zeros(1, self.num_classes))
        
        self.apply(initialize_weights)
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        
        # Feature transformation
        h = self.feature(x)  # [B, N, L] or [N, L]
        h = h.squeeze(0)  # [N, L]
        
        # Instance-level scoring
        scores = self.scoring(h)  # [N, num_classes]
        
        # Additive aggregation: sum instance scores and add bias
        logits = scores.sum(dim=0, keepdim=True) + self.bias  # [1, num_classes]
        
        forward_return['logits'] = logits
        
        if return_WSI_feature:
            # Use mean of instance features as bag-level feature
            forward_return['WSI_feature'] = h.mean(dim=0, keepdim=True)
        
        if return_WSI_attn:
            # Instance scores can be interpreted as importance/attention
            # Normalize scores to get attention-like weights
            forward_return['WSI_attn'] = scores
        
        return forward_return
