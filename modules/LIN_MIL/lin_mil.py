"""Lin-MIL linear-attention aggregator.

Ported from https://github.com/charlotterchtr/Lin-MIL (MIT license).
"""

import torch
import torch.nn as nn
from einops import repeat
from timm.layers import DropPath


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, act_layer=nn.GELU):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class LinearAttention(nn.Module):
    """Official ReLU-kernel attention with O(ND^2) complexity."""

    def __init__(self, dim, num_heads=8, qkv_bias=True, att_act=nn.ReLU):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.num_heads = num_heads
        head_dim = dim // num_heads
        scale = head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * scale)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.act = att_act()

    def forward(self, x):
        batch, instances, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            instances,
            3,
            self.num_heads,
            channels // self.num_heads,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = self.act(q)
        k = self.act(k)
        denominator = torch.clamp(
            q @ k.transpose(-2, -1).sum(dim=-1, keepdim=True),
            min=1e2,
        )
        channel_attention = (k.transpose(-2, -1) @ v) * self.temperature
        unnormalized_attention = q @ channel_attention
        attended = unnormalized_attention / denominator
        attended = attended.transpose(1, 2).reshape(
            batch, instances, channels
        )
        return self.proj(attended), unnormalized_attention


class LinearTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        att_act=nn.ReLU,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            att_act=att_act,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        attended, attention = self.attn(self.norm1(x))
        x = x + self.drop_path(attended)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attention


def _activation(name):
    activations = {
        "GeLU": nn.GELU,
        "ReLU": nn.ReLU,
        "ELU": nn.ELU,
        "LeakyReLU": nn.LeakyReLU,
        "TanH": nn.Tanh,
        "Softplus": nn.Softplus,
    }
    if name not in activations:
        raise ValueError(f"Invalid Lin-MIL activation: {name}")
    return activations[name]


class LIN_MIL(nn.Module):
    def __init__(
        self,
        in_dim,
        num_classes,
        latent_dim=512,
        transformer_depth=4,
        dropout=0.0,
        emb_dropout=0.1,
        act="ReLU",
        pooling="cls",
        num_heads=8,
        max_instances=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.pooling = pooling
        self.max_instances = max_instances
        self.projection = nn.Linear(in_dim, latent_dim, bias=True)
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = nn.ModuleList(
            [
                LinearTransformerBlock(
                    dim=latent_dim,
                    num_heads=num_heads,
                    qkv_bias=True,
                    drop_path=dropout,
                    att_act=_activation(act),
                )
                for _ in range(transformer_depth)
            ]
        )
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        elif pooling != "mean":
            raise ValueError("LIN_MIL pooling must be 'cls' or 'mean'")
        self.norm = nn.LayerNorm(latent_dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(latent_dim), nn.Linear(latent_dim, num_classes)
        )

    def _select_instances(self, x, apply_instance_limit):
        num_instances = x.shape[1]
        if (
            not apply_instance_limit
            or self.max_instances is None
            or num_instances <= self.max_instances
        ):
            return x, torch.arange(num_instances, device=x.device)
        if self.training:
            indices = torch.randperm(num_instances, device=x.device)[
                : self.max_instances
            ].sort().values
        else:
            indices = torch.linspace(
                0,
                num_instances - 1,
                steps=self.max_instances,
                device=x.device,
            ).round().long()
        return x.index_select(1, indices), indices

    def forward(
        self,
        x,
        return_WSI_feature=False,
        return_WSI_attn=False,
        apply_instance_limit=True,
        **_,
    ):
        input_was_2d = x.dim() == 2
        if input_was_2d:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError("LIN_MIL expects [N, D] or [B, N, D]")
        original_instances = x.shape[1]
        x, indices = self._select_instances(x, apply_instance_limit)
        x = self.projection(x)
        if self.pooling == "cls":
            cls_tokens = repeat(
                self.cls_token, "1 1 d -> b 1 d", b=x.shape[0]
            )
            x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        attention = None
        for layer in self.transformer:
            x, attention = layer(x)
        importance = attention.mean(dim=1).mean(dim=-1)

        if self.pooling == "mean":
            bag_feature = x.mean(dim=1)
        else:
            bag_feature = x[:, 0]
            importance = importance[:, 1:]
        bag_feature = self.norm(bag_feature)
        logits = self.mlp_head(bag_feature)

        full_importance = importance.new_zeros(
            importance.shape[0], original_instances
        )
        full_importance.index_copy_(1, indices, importance)
        wsi_attention = full_importance.unsqueeze(-1)
        if input_was_2d:
            wsi_attention = wsi_attention.squeeze(0)

        result = {"logits": logits}
        if return_WSI_feature:
            result["WSI_feature"] = bag_feature
        if return_WSI_attn:
            # Official importance: final-layer attention averaged over heads/dim.
            result["WSI_attn"] = wsi_attention
        return result
