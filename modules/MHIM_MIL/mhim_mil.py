"""
MHIM_MIL - Multi-Head Instance Masking Multiple Instance Learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class Attention(nn.Module):
    """Simple attention mechanism"""
    def __init__(self, input_dim=512, act='relu', bias=False, dropout=False):
        super(Attention, self).__init__()
        self.L = input_dim
        self.D = 128
        self.K = 1

        self.attention = [nn.Linear(self.L, self.D, bias=bias)]

        if act == 'gelu':
            self.attention += [nn.GELU()]
        elif act == 'relu':
            self.attention += [nn.ReLU()]
        elif act == 'tanh':
            self.attention += [nn.Tanh()]

        if dropout:
            self.attention += [nn.Dropout(0.25)]

        self.attention += [nn.Linear(self.D, self.K, bias=bias)]
        self.attention = nn.Sequential(*self.attention)

    def forward(self, x, no_norm=False):
        A = self.attention(x)
        A = torch.transpose(A, -1, -2)  # KxN
        A_ori = A.clone()
        A = F.softmax(A, dim=-1)  # softmax over N
        x = torch.matmul(A, x)
        
        if no_norm:
            return x, A_ori
        else:
            return x, A

class SAttention(nn.Module):
    """Self-attention with masking support"""
    def __init__(self, mlp_dim=512, head=8, dropout=0.1):
        super(SAttention, self).__init__()
        self.norm = nn.LayerNorm(mlp_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=mlp_dim,
            num_heads=head,
            dropout=dropout,
            batch_first=True
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, mlp_dim))

    def forward(self, x, mask_ids=None, len_keep=None, return_attn=False, mask_enable=False):
        batch, num_patches, C = x.shape
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply masking if enabled
        if mask_enable and mask_ids is not None:
            # mask_ids shape: (1, N) where first len_keep are kept
            mask_ids = mask_ids.squeeze(0)  # (N,)
            # Add 1 for CLS token
            mask_ids = mask_ids + 1
            mask_ids = torch.cat([torch.tensor([0], device=x.device), mask_ids])  # Keep CLS token
            x = x[:, mask_ids, :]
        
        x = self.norm(x)
        attn_output, attn_weights = self.multihead_attn(x, x, x, need_weights=return_attn)
        
        # Extract CLS token
        cls_feat = attn_output[:, 0, :]  # (batch, mlp_dim)
        
        if return_attn:
            return cls_feat, attn_weights
        else:
            return cls_feat

class MHIM_MIL(nn.Module):
    """
    Multi-Head Instance Masking Multiple Instance Learning
    Uses self-attention with instance masking mechanism
    """
    def __init__(self, in_dim=1024, num_classes=2, dropout=0.25, act=nn.ReLU(), 
                 mlp_dim=512, head=8, mask_ratio=0.0, baseline='selfattn', **kwargs):
        super(MHIM_MIL, self).__init__()
        
        self.mask_ratio = mask_ratio
        self.baseline = baseline
        
        # Convert act to string
        if isinstance(act, nn.ReLU):
            act_str = 'relu'
        elif isinstance(act, nn.GELU):
            act_str = 'gelu'
        else:
            act_str = 'relu'
        
        # Patch to embedding
        self.patch_to_emb = [nn.Linear(in_dim, mlp_dim)]
        if act_str.lower() == 'relu':
            self.patch_to_emb += [nn.ReLU()]
        elif act_str.lower() == 'gelu':
            self.patch_to_emb += [nn.GELU()]
        
        self.dp = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.patch_to_emb = nn.Sequential(*self.patch_to_emb)
        
        # Encoder
        if baseline == 'selfattn':
            self.online_encoder = SAttention(mlp_dim=mlp_dim, head=head, dropout=dropout)
        elif baseline == 'attn':
            self.online_encoder = Attention(input_dim=mlp_dim, act=act_str, dropout=dropout)
        else:
            self.online_encoder = Attention(input_dim=mlp_dim, act=act_str, dropout=dropout)
        
        # Predictor
        self.predictor = nn.Linear(mlp_dim, num_classes)
        
        self.apply(initialize_weights)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        forward_return = {}
        
        # Handle input format
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)
        
        # Patch to embedding
        x = self.patch_to_emb(x)  # (1, N, mlp_dim)
        x = self.dp(x)
        
        ps = x.size(1)  # number of patches
        
        # Apply masking if training and mask_ratio > 0
        mask_ids = None
        len_keep = ps
        if self.training and self.mask_ratio > 0.0:
            len_keep = int(ps * (1 - self.mask_ratio))
            # Random masking
            ids_shuffle = torch.randperm(ps, device=x.device).unsqueeze(0)
            mask_ids = ids_shuffle[:, len_keep:]
            len_keep = len_keep
        
        # Encode
        if self.baseline == 'selfattn':
            if return_WSI_attn:
                cls_feat, attn_weights = self.online_encoder(
                    x, mask_ids=mask_ids, len_keep=len_keep, 
                    return_attn=True, mask_enable=(mask_ids is not None)
                )
            else:
                cls_feat = self.online_encoder(
                    x, mask_ids=mask_ids, len_keep=len_keep, 
                    return_attn=False, mask_enable=(mask_ids is not None)
                )
                attn_weights = None
        else:
            if mask_ids is not None:
                # Apply masking
                mask_ids = mask_ids.squeeze(0)
                x = x[:, mask_ids, :]
            cls_feat, attn_weights = self.online_encoder(x, no_norm=not return_WSI_attn)
        
        # Prediction
        logits = self.predictor(cls_feat)  # (1, num_classes)
        
        forward_return['logits'] = logits
        if return_WSI_feature:
            forward_return['WSI_feature'] = cls_feat
        if return_WSI_attn:
            if attn_weights is not None:
                # Extract attention from multihead attention
                if isinstance(attn_weights, tuple):
                    attn_weights = attn_weights[0]  # Take first element if tuple
                if len(attn_weights.shape) == 3:
                    # Average over heads: (batch, num_patches+1, num_patches+1)
                    attn_weights = attn_weights.mean(dim=1)  # Average over heads
                    # Extract attention to CLS token (first token)
                    attn_weights = attn_weights[0, 1:]  # Remove CLS token itself, get attention to patches
                forward_return['WSI_attn'] = attn_weights
            else:
                # Fallback: uniform attention
                forward_return['WSI_attn'] = torch.ones(ps, device=x.device) / ps
        
        return forward_return
