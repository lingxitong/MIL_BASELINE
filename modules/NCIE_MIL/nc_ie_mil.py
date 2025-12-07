import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class NcIEMIL(nn.Module):
    def __init__(self, in_dim, in_chans, latent_dim=1024, n_classes=2, num_heads=4, ratio=32,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., conv_drop=0., mode='cross'):
        super(NcIEMIL, self).__init__()
        assert in_dim % num_heads == 0, f"dim {in_dim} should be divided by num_heads {num_heads}."

        self.in_dim = in_dim
        self.in_chans = in_chans
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.h = self.w = int(math.sqrt(in_chans))
        self.mode = mode
        assert self.h * self.w == self.in_chans, f"in_channs {in_chans} should be a perfect square number."

        self.num_heads = num_heads
        self.dim_per_head = latent_dim // self.num_heads
        self.scale = self.dim_per_head ** -0.5

        self.embed = nn.Sequential(nn.Linear(in_dim, latent_dim), nn.ReLU(), nn.LayerNorm(latent_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))    
        self.c_attn = ChannelAttention(dim=latent_dim, ratio=ratio, conv_drop=conv_drop)
        self.s_attn = SpatialAttention(dim=latent_dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.norm3 = nn.LayerNorm(latent_dim)

        self._q = nn.Linear(latent_dim, latent_dim, bias=qkv_bias)
        self._kv = nn.Linear(latent_dim, latent_dim, bias=qkv_bias)
        self._attn_drop = nn.Dropout(attn_drop)
        self._proj = nn.Linear(latent_dim, latent_dim)
        self._proj_drop = nn.Dropout(proj_drop)
        self.classifier = nn.Linear(latent_dim, n_classes)

        trunc_normal_(self.cls_token, std=.02)
        self.apply(initialize_weights)

    def forward(self, h, return_WSI_attn=False, return_WSI_feature=False):
        """
        Forward pass for NcIEMIL model
        
        Args:
            h: input features [B, N, in_dim] or [N, in_dim] (will be unsqueezed)
            return_WSI_attn: whether to return attention weights
            return_WSI_feature: whether to return WSI-level features
            
        Returns:
            forward_return: dictionary containing:
                - 'logits': classification logits
                - 'WSI_attn': attention weights (if return_WSI_attn=True)
                - 'WSI_feature': WSI-level features (if return_WSI_feature=True)
        """
        forward_return = {}
        
        # Handle input shape: if 2D, add batch dimension
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        
        h = self.embed(h)
        B, _, C = h.shape

        cls_token = self.cls_token.expand(B, -1, -1)
        h_s = torch.cat((cls_token, h), dim=1)

        h_s = h_s + self.s_attn(self.norm1(h_s))
        h_c = h + self.c_attn(self.norm2(h), self.h, self.w)

        _q = self._q(h_s).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        _kv = self._kv(h_c).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        _k, _v = _kv[0], _kv[1]

        _attn = (_q @ _k.transpose(-2, -1)) * self.scale
        _attn = _attn.softmax(dim=-1)
        _attn = self._attn_drop(_attn)

        h = (_attn @ _v).transpose(1, 2).reshape(B, -1, C)
        h = self._proj(h)
        h = self._proj_drop(h)
        
        # Extract CLS token
        h_cls = self.norm3(h)[:, 0]
        
        # Store attention weights if needed
        if return_WSI_attn:
            # Extract attention weights for CLS token (first token)
            forward_return['WSI_attn'] = _attn[:, :, 0, 1:].squeeze(1)  # Remove CLS token from attention
        
        logits = self.classifier(h_cls)
        forward_return['logits'] = logits
        
        if return_WSI_feature:
            forward_return['WSI_feature'] = h_cls

        return forward_return


class ChannelAttention(nn.Module):
    def __init__(self, dim, ratio=16, conv_drop=0.):
        super(ChannelAttention, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=1, padding=1, groups=dim)
        self.conv_drop = nn.Dropout(conv_drop)

        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // ratio, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(dim // ratio),
            nn.GELU(),
            nn.Conv2d(dim // ratio, dim, kernel_size=(1, 1), bias=False),
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W, f"Height {H} and width {W} do not match spatial dimension {N}."

        x = x.reshape(B, H, -1, C).permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = self.conv_drop(x)
        v = self.pooling(x)
        v = self.fc(v)
        v = self.sigmoid(v)
        x = (v * x).flatten(2).transpose(1, 2)

        return x


class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1 ,3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k ,v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

