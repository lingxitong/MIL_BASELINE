"""Anchor-stabilized attention MIL.

Adapted from the official ASMIL implementation accompanying the ICLR 2026
paper: https://github.com/Linfeng-Ye/ASMIL.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_sigmoid(x, dim=-1, eps=1e-8):
    values = torch.sigmoid(x)
    return values / values.sum(dim=dim, keepdim=True).clamp_min(eps)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def _split_heads(self, x):
        batch_size, tokens, channels = x.shape
        return x.view(
            batch_size, tokens, self.num_heads, channels // self.num_heads
        ).transpose(1, 2)

    def forward(self, query, key, value, normalization="softmax"):
        query = self._split_heads(self.q_proj(query))
        key = self._split_heads(self.k_proj(key))
        value = self._split_heads(self.v_proj(value))
        raw_attention = torch.matmul(query, key.transpose(-1, -2))
        raw_attention = raw_attention / math.sqrt(self.head_dim)
        if normalization == "softmax":
            attention = F.softmax(raw_attention, dim=-1)
        elif normalization == "sigmoid":
            attention = normalized_sigmoid(raw_attention, dim=-1)
        else:
            raise ValueError("normalization must be 'softmax' or 'sigmoid'")
        output = torch.matmul(attention, value).transpose(1, 2).contiguous()
        output = output.flatten(2)
        return self.norm(self.dropout(self.out_proj(output))), raw_attention


class AttentionTokenMIL(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        num_tokens,
        num_heads,
        token_drop,
        dropout,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_drop = token_drop
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
        )
        self.token_attention = nn.ModuleList(
            [
                MultiHeadCrossAttention(hidden_dim, num_heads, dropout)
                for _ in range(num_tokens)
            ]
        )
        self.queries = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.queries, std=1e-6)
        self.token_classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, num_classes) for _ in range(num_tokens)]
        )
        self.bag_attention = MultiHeadCrossAttention(
            hidden_dim, num_heads, dropout
        )
        self.slide_classifier = nn.Linear(hidden_dim, num_classes)

    def attention_scores(self, x):
        features = self.projection(x)
        raw_scores = []
        for index, attention in enumerate(self.token_attention):
            _, raw = attention(
                self.queries[:, index : index + 1].expand(x.shape[0], -1, -1),
                features,
                features,
            )
            raw_scores.append(raw)
        return torch.cat(raw_scores, dim=2)

    def forward(self, x):
        features = self.projection(x)
        token_features = []
        token_logits = []
        raw_scores = []
        for index, attention in enumerate(self.token_attention):
            token_feature, raw = attention(
                self.queries[:, index : index + 1].expand(x.shape[0], -1, -1),
                features,
                features,
            )
            token_features.append(token_feature)
            token_logits.append(
                self.token_classifiers[index](token_feature.squeeze(1))
            )
            raw_scores.append(raw)

        tokens = torch.cat(token_features, dim=1)
        if self.training and self.token_drop > 0:
            keep_count = self.num_tokens - self.token_drop
            keep = torch.randperm(self.num_tokens, device=x.device)[:keep_count]
            keep = keep.sort().values
            tokens_for_bag = tokens.index_select(1, keep)
        else:
            tokens_for_bag = tokens
        bag_feature, _ = self.bag_attention(
            self.cls_token.expand(x.shape[0], -1, -1),
            tokens_for_bag,
            tokens_for_bag,
        )
        bag_feature = bag_feature.squeeze(1)
        return {
            "logits": self.slide_classifier(bag_feature),
            "sub_logits": torch.stack(token_logits, dim=1),
            "raw_attention": torch.cat(raw_scores, dim=2),
            "bag_feature": bag_feature,
        }


class AS_MIL(nn.Module):
    """Attention MIL trained against a normalized-sigmoid EMA anchor."""

    def __init__(
        self,
        in_dim=512,
        hidden_dim=256,
        num_classes=2,
        num_tokens=8,
        num_heads=8,
        token_drop=4,
        dropout=0.1,
        ema_decay=0.999,
        temperature=0.2,
        consistency_weight=1.0,
    ):
        super().__init__()
        if num_tokens < 1 or not 0 <= token_drop < num_tokens:
            raise ValueError("token_drop must be in [0, num_tokens)")
        self.in_dim = in_dim
        self.ema_decay = ema_decay
        self.temperature = temperature
        self.consistency_weight = consistency_weight
        self.online = AttentionTokenMIL(
            in_dim,
            hidden_dim,
            num_classes,
            num_tokens,
            num_heads,
            token_drop,
            dropout,
        )
        self.anchor = copy.deepcopy(self.online)
        self.anchor.token_drop = 0
        for parameter in self.anchor.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def update_anchor(self):
        online_state = self.online.state_dict()
        anchor_state = self.anchor.state_dict()
        for name, anchor_value in anchor_state.items():
            online_value = online_state[name]
            if torch.is_floating_point(anchor_value):
                anchor_value.mul_(self.ema_decay).add_(
                    online_value, alpha=1.0 - self.ema_decay
                )
            else:
                anchor_value.copy_(online_value)

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
                f"AS_MIL expects [N, {self.in_dim}] or [B, N, {self.in_dim}]"
            )

        output = self.online(x)
        online_attention = F.softmax(output["raw_attention"], dim=-1)
        result = {
            "logits": output["logits"],
            "sub_logits": output["sub_logits"],
            "raw_attention": output["raw_attention"],
        }
        if self.training:
            with torch.no_grad():
                anchor_raw = self.anchor.attention_scores(x)
                anchor_attention = normalized_sigmoid(
                    anchor_raw / self.temperature, dim=-1
                )
            consistency_loss = -torch.nan_to_num(
                anchor_attention * online_attention.clamp_min(1e-8).log()
            ).mean()
            result["consistency_loss"] = (
                self.consistency_weight * consistency_loss
            )
        if return_WSI_feature:
            result["WSI_feature"] = output["bag_feature"]
        if return_WSI_attn:
            importance = online_attention.mean(dim=(1, 2)).unsqueeze(-1)
            result["WSI_attn"] = importance.squeeze(0) if input_was_2d else importance
        return result
