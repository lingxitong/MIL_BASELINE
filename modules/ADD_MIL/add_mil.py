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

class ADD_MIL(nn.Module):
    """
    Additive MIL: Intrinsically Interpretable MIL for Pathology (NeurIPS 2022)
    Key idea: Uses additive scoring mechanism where instance-level predictions
    are aggregated additively for better interpretability.
    """
    def __init__(self, L=512, D=128, num_classes=2, dropout=0, act=nn.ReLU(), in_dim=512, rrt=None):
        super(ADD_MIL, self).__init__()
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
        
        # Attention mechanism for weighting instances
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, 1)
        )
        
        # Instance-level classifier
        self.instance_classifier = nn.Sequential(
            nn.Linear(self.L, self.num_classes)
        )
        
        self.apply(initialize_weights)
    
    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        
        # Handle input format - support both 2D and 3D inputs
        if len(x.shape) == 2:
            # Input is (N, D) - already 2D
            x = x.unsqueeze(0)  # (1, N, D)
        elif len(x.shape) == 3:
            # Input is (1, N, D) or (B, N, D)
            pass  # Already 3D
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # Extract features - feature layer expects (B, N, D) or (N, D)
        # Reshape to (B*N, D) for feature extraction, then reshape back
        batch_size, num_instances, feat_dim = x.shape
        x_flat = x.view(-1, feat_dim)  # (B*N, D)
        feature_flat = self.feature(x_flat)  # (B*N, L)
        feature = feature_flat.view(batch_size, num_instances, -1)  # (B, N, L)
        feature = feature.squeeze(0)  # [N, L]
        
        # Compute attention weights
        A = self.attention(feature)  # NxK where K=1
        A = torch.transpose(A, -1, -2)  # 1xN
        A = F.softmax(A, dim=-1)  # Attention weights
        
        # Compute instance-level predictions
        instance_preds = self.instance_classifier(feature)  # Nx num_classes
        
        # Additive aggregation: weighted sum of instance predictions
        logits = torch.mm(A, instance_preds)  # 1 x num_classes
        
        forward_return['logits'] = logits
        
        if return_WSI_feature:
            # Weighted average of features
            M = torch.mm(A, feature)
            forward_return['WSI_feature'] = M
        
        if return_WSI_attn:
            forward_return['WSI_attn'] = A.transpose(-1, -2)
        
        return forward_return
