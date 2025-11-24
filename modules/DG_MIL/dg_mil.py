"""
DG_MIL - Distribution-Guided Multiple Instance Learning
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
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

class DG_MIL(nn.Module):
    """
    Distribution-Guided Multiple Instance Learning
    Uses projection head and distribution-guided attention
    """
    def __init__(self, in_dim=1024, num_classes=2, dropout=0.1, act=nn.ReLU(), 
                 projection_dim=768, **kwargs):
        super(DG_MIL, self).__init__()
        
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.projection_dim = projection_dim
        
        # Projection head (similar to Linear_projection_MAE but adaptable)
        if projection_dim == 768:
            self.projection_head = nn.Linear(in_dim, projection_dim)
        elif projection_dim == 512:
            self.projection_head = nn.Linear(in_dim, projection_dim)
        elif projection_dim == 256:
            self.projection_head = nn.Linear(in_dim, projection_dim)
        else:
            self.projection_head = nn.Linear(in_dim, projection_dim)
        
        self.bn = nn.BatchNorm1d(projection_dim)
        
        # Attention mechanism for bag-level aggregation
        self.attention = nn.Sequential(
            nn.Linear(projection_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.classifier = nn.Linear(projection_dim, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # Convert act to appropriate activation
        if isinstance(act, nn.ReLU):
            self.act = nn.ReLU()
        elif isinstance(act, nn.GELU):
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()
        
        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        """
        前向传播
        输入: x - (1, N, D) 形状的tensor
        输出: forward_return - 包含 'logits' 的字典
        """
        forward_return = {}
        
        # Handle input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        instances = x.squeeze(0)  # (N, D)
        
        # Project features
        projected = self.projection_head(instances)  # (N, projection_dim)
        
        # BatchNorm (requires 2D input)
        if projected.shape[0] > 1:
            projected = self.bn(projected)
        else:
            # For single instance, skip batch norm or use instance norm
            pass
        
        projected = self.act(projected)
        projected = self.dropout(projected)
        
        # Attention-based aggregation
        A = self.attention(projected)  # (N, 1)
        A_ori = A.clone()
        A = torch.transpose(A, -1, -2)  # (1, N)
        A = F.softmax(A, dim=-1)  # (1, N)
        
        # Aggregate features
        wsi_feature = torch.mm(A, projected)  # (1, projection_dim)
        
        # Classification
        logits = self.classifier(wsi_feature)  # (1, num_classes)
        
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = wsi_feature
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori.squeeze(-1)  # (N,)
        
        return forward_return

