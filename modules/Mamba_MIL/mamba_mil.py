"""
MambaMIL - Mamba-based Multiple Instance Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Try importing from local mamba package first (from MambaMIL project)
    try:
        from mamba.mamba_ssm.modules.srmamba import SRMamba
        from mamba.mamba_ssm.modules.bimamba import BiMamba
        from mamba.mamba_ssm.modules.mamba_simple import Mamba
    except ImportError:
        # Fallback to standard mamba-ssm package
        try:
            from mamba_ssm.modules.mamba_simple import Mamba
            # For SRMamba and BiMamba, we may need to implement them or use Mamba
            SRMamba = BiMamba = Mamba
        except ImportError:
            raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm")
except ImportError as e:
    print(f"Warning: mamba-ssm import failed: {e}")
    print("Please ensure mamba-ssm is installed or the local mamba package is available")
    SRMamba = BiMamba = Mamba = None

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Mamba_MIL(nn.Module):
    """
    Mamba-based Multiple Instance Learning
    Reference: MambaMIL paper
    """
    def __init__(self, in_dim=1024, num_classes=2, dropout=0.1, act=nn.ReLU(), 
                 layer=2, rate=10, mamba_type="SRMamba", **kwargs):
        super(Mamba_MIL, self).__init__()
        
        # Convert act to string for compatibility
        if isinstance(act, nn.ReLU):
            act_str = 'relu'
        elif isinstance(act, nn.GELU):
            act_str = 'gelu'
        else:
            act_str = 'relu'
        
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act_str.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act_str.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()
        self.mamba_type = mamba_type
        self.rate = rate

        if SRMamba is None:
            raise ImportError("mamba-ssm is required. Install with: pip install mamba-ssm")

        if mamba_type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        SRMamba(
                            d_model=512,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        elif mamba_type == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        Mamba(
                            d_model=512,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        elif mamba_type == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    )
                )
        else:
            raise NotImplementedError(f"Mamba type [{mamba_type}] is not implemented")

        self.num_classes = num_classes
        self.classifier = nn.Linear(512, self.num_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

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
        h = x.float()  # [B, n, in_dim]
        
        h = self._fc1(h)  # [B, n, 512]

        # Apply Mamba layers
        if self.mamba_type == "SRMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)  # LayerNorm
                # SRMamba forward signature may vary, try with rate first
                try:
                    h = layer[1](h, rate=self.rate)  # SRMamba with rate
                except TypeError:
                    # If rate is not supported, call without it
                    h = layer[1](h)
                h = h + h_  # Residual connection
        elif self.mamba_type == "Mamba" or self.mamba_type == "BiMamba":
            for layer in self.layers:
                h_ = h
                h = layer[0](h)  # LayerNorm
                h = layer[1](h)  # Mamba/BiMamba
                h = h + h_  # Residual connection

        h = self.norm(h)
        A = self.attention(h)  # [B, n, 1]
        A_ori = A.clone()
        A = torch.transpose(A, 1, 2)  # [B, 1, n]
        A = F.softmax(A, dim=-1)  # [B, 1, n]
        h = torch.bmm(A, h)  # [B, 1, 512]
        h = h.squeeze(0)  # [1, 512]

        logits = self.classifier(h)  # [1, num_classes]
        
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = h
        if return_WSI_attn:
            forward_return['WSI_attn'] = A_ori.squeeze(0).squeeze(-1)  # [n]
        
        return forward_return
