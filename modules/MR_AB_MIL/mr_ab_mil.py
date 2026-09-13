"""MR-ABMIL with manifold residual attention projections.

Adapted from https://github.com/BearCleverProud/MR-Block (MIT license),
the official implementation accompanying the ICLR 2026 MR-Block paper.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankResidualPath(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0, dropout=0.0):
        super().__init__()
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.down_projection = nn.Parameter(torch.empty(rank, in_features))
        self.up_projection = nn.Parameter(torch.empty(out_features, rank))
        self.activation = nn.GELU()
        nn.init.kaiming_uniform_(self.down_projection, a=math.sqrt(5))
        nn.init.zeros_(self.up_projection)

    def forward(self, x):
        hidden = F.linear(self.dropout(x), self.down_projection)
        return F.linear(self.activation(hidden), self.up_projection) * self.scaling


class ManifoldResidualBlock(nn.Linear):
    """Frozen affine map plus a trainable nonlinear low-rank residual path."""

    def __init__(
        self,
        in_features,
        out_features,
        rank=64,
        residual_alpha=1.0,
        residual_dropout=0.0,
        bias=True,
    ):
        if rank < 1:
            raise ValueError("rank must be positive")
        super().__init__(in_features, out_features, bias=bias)
        self.residual_path = LowRankResidualPath(
            in_features,
            out_features,
            rank,
            alpha=residual_alpha,
            dropout=residual_dropout,
        )
        self.weight.requires_grad = False

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.residual_path(x)


class MR_AB_MIL(nn.Module):
    """Official MR-ABMIL variant with gated attention and MR projections."""

    def __init__(
        self,
        in_dim=512,
        attention_dim=256,
        num_classes=2,
        rank=64,
        residual_alpha=1.0,
        residual_dropout=0.0,
        dropout=0.0,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.attention_a = nn.Sequential(
            ManifoldResidualBlock(
                in_dim,
                attention_dim,
                rank=rank,
                residual_alpha=residual_alpha,
                residual_dropout=residual_dropout,
            ),
            nn.Tanh(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.attention_b = nn.Sequential(
            ManifoldResidualBlock(
                in_dim,
                attention_dim,
                rank=rank,
                residual_alpha=residual_alpha,
                residual_dropout=residual_dropout,
            ),
            nn.Sigmoid(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.attention_c = nn.Linear(attention_dim, 1)
        self.classifier = nn.Linear(in_dim, num_classes)
        nn.init.xavier_normal_(self.attention_c.weight)
        nn.init.zeros_(self.attention_c.bias)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        x,
        return_WSI_feature=False,
        return_WSI_attn=False,
        **_,
    ):
        input_was_2d = x.dim() == 2
        if input_was_2d:
            x = x.unsqueeze(0)
        if x.dim() != 3 or x.shape[-1] != self.in_dim:
            raise ValueError(
                f"MR_AB_MIL expects [N, {self.in_dim}] or [B, N, {self.in_dim}]"
            )

        raw_attention = self.attention_c(
            self.attention_a(x) * self.attention_b(x)
        ).squeeze(-1)
        attention = F.softmax(raw_attention, dim=-1)
        bag_feature = torch.bmm(attention.unsqueeze(1), x).squeeze(1)
        logits = self.classifier(bag_feature)

        result = {
            "logits": logits,
            "raw_attention": raw_attention,
            "attention": attention,
        }
        if return_WSI_feature:
            result["WSI_feature"] = bag_feature
        if return_WSI_attn:
            wsi_attention = raw_attention.unsqueeze(-1)
            result["WSI_attn"] = (
                wsi_attention.squeeze(0) if input_was_2d else wsi_attention
            )
        return result
