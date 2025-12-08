"""
MAMBA2D_MIL - 2D Mamba Multiple Instance Learning
Reference: https://github.com/AtlasAnalyticsLab/2DMamba
Paper: 2DMamba: Efficient State Space Model for Image Representation (CVPR 2025)

Note: This implementation provides a simplified 2D Mamba approach for WSI MIL.
For full 2D Mamba functionality, refer to the original repository.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba.mamba_ssm.modules.mamba_simple import Mamba
    from mamba.mamba_ssm.modules.srmamba import SRMamba
    MAMBA_AVAILABLE = True
except ImportError:
    try:
        from mamba_ssm.modules.mamba_simple import Mamba
        SRMamba = Mamba
        MAMBA_AVAILABLE = True
    except ImportError:
        MAMBA_AVAILABLE = False
        print("Warning: mamba-ssm is required. Install with: pip install mamba-ssm")

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Mamba2DBlock(nn.Module):
    """
    2D Mamba Block - applies Mamba along spatial dimensions
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required for MAMBA2D_MIL")
        
        # Mamba layers for row and column processing
        self.mamba_row = SRMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mamba_col = SRMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """
        Args:
            x: (B, H, W, D) or (H*W, D) - spatial feature map
        Returns:
            out: same shape as input
        """
        if len(x.shape) == 2:
            # Flattened input, need to reshape (assume square grid)
            N, D = x.shape
            H = W = int(N ** 0.5)
            x = x.view(1, H, W, D)
        
        B, H, W, D = x.shape
        
        # Process rows
        x_row = x.view(B * H, W, D)  # (B*H, W, D)
        # Check if mamba supports rate parameter
        try:
            x_row = self.mamba_row(x_row, rate=10)  # Try with rate parameter
        except TypeError:
            x_row = self.mamba_row(x_row)  # Fallback without rate
        x_row = x_row.view(B, H, W, D)
        
        # Process columns
        x_col = x_row.permute(0, 2, 1, 3).contiguous()  # (B, W, H, D)
        x_col = x_col.view(B * W, H, D)  # (B*W, H, D)
        try:
            x_col = self.mamba_col(x_col, rate=10)  # Try with rate parameter
        except TypeError:
            x_col = self.mamba_col(x_col)  # Fallback without rate
        x_col = x_col.view(B, W, H, D)
        x_col = x_col.permute(0, 2, 1, 3).contiguous()  # (B, H, W, D)
        
        # Residual and norm
        out = self.norm(x_col + x)
        return out

class Mamba2D_MIL(nn.Module):
    """
    2D Mamba Multiple Instance Learning
    
    Uses 2D Mamba for spatial modeling of patch features arranged in a grid.
    Requires patch coordinates to reconstruct 2D spatial layout.
    
    Reference: https://github.com/AtlasAnalyticsLab/2DMamba
    """
    def __init__(self, in_dim=1024, num_classes=2, dropout=0.1, act=nn.ReLU(), 
                 d_model=512, d_state=16, n_layers=2, grid_size=None):
        super(Mamba2D_MIL, self).__init__()
        
        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm")
        
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.d_model = d_model
        self.grid_size = grid_size
        
        # Convert act to string
        if isinstance(act, nn.ReLU):
            act_str = 'relu'
        elif isinstance(act, nn.GELU):
            act_str = 'gelu'
        else:
            act_str = 'relu'
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU() if act_str.lower() == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2D Mamba layers
        self.mamba2d_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mamba2d_layers.append(
                Mamba2DBlock(d_model=d_model, d_state=d_state)
            )
        
        # Aggregation
        self.attention = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.classifier = nn.Linear(d_model, num_classes)
        
        self.apply(initialize_weights)
    
    def _features_to_grid(self, features, coords=None):
        """
        Convert patch features to 2D grid
        Args:
            features: (N, D) patch features
            coords: (N, 2) patch coordinates (optional)
        Returns:
            grid: (H, W, D) or flattened (H*W, D)
        """
        N, D = features.shape
        
        if coords is not None:
            # Use provided coordinates
            coords_norm = (coords - coords.min(0)[0]) / (coords.max(0)[0] - coords.min(0)[0] + 1e-8)
            H = int(coords_norm[:, 0].max().item() * 100) + 1
            W = int(coords_norm[:, 1].max().item() * 100) + 1
            grid = torch.zeros(H, W, D, device=features.device)
            indices = (coords_norm * torch.tensor([H-1, W-1], device=features.device)).long()
            grid[indices[:, 0], indices[:, 1]] = features
        else:
            # Assume square grid
            H = W = int(N ** 0.5)
            if H * W < N:
                # Need larger grid
                H = W = int(N ** 0.5) + 1
            if H * W != N:
                # Pad to square if needed
                pad_size = H * W - N
                if pad_size > 0:
                    padding = torch.zeros(pad_size, D, device=features.device)
                    features = torch.cat([features, padding], dim=0)
                elif pad_size < 0:
                    # If still not enough, truncate (shouldn't happen with above logic)
                    features = features[:H*W]
            grid = features.view(H, W, D)
        
        return grid
    
    def forward(self, x, coords=None, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        
        # Handle input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        x = x.squeeze(0)  # (N, D)
        N = x.shape[0]
        
        # Feature projection
        x = self.feature_proj(x)  # (N, d_model)
        
        # Convert to 2D grid
        grid = self._features_to_grid(x, coords)  # (H, W, d_model)
        H, W, D = grid.shape
        
        # Apply 2D Mamba layers
        h = grid.unsqueeze(0)  # (1, H, W, D)
        for mamba2d_layer in self.mamba2d_layers:
            h = mamba2d_layer(h)  # Mamba2DBlock handles rate internally if needed
        
        h = h.squeeze(0)  # (H, W, D)
        h_flat = h.view(H * W, D)  # (H*W, D)
        
        # Attention aggregation
        A = self.attention(h_flat)  # (H*W, 1)
        A_ori = A.clone()
        A = A.transpose(0, 1)  # (1, H*W)
        A = F.softmax(A, dim=-1)  # (1, H*W)
        bag_feature = torch.mm(A, h_flat)  # (1, D)
        # Keep batch dimension for training compatibility
        # bag_feature shape: (1, D)
        
        # Classification
        logits = self.classifier(bag_feature)  # (1, num_classes)
        
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = bag_feature.unsqueeze(0)
        if return_WSI_attn:
            # Map attention back to original patches
            if N <= H * W:
                attn = A_ori[:N].squeeze(-1)
            else:
                attn = A_ori.squeeze(-1)
            forward_return['WSI_attn'] = attn
        
        return forward_return

