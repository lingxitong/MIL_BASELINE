"""FC-MIL, adapted from the authors' Apache-2.0 implementation.

Source: https://github.com/7FFDW/FCMIL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicTanhBlock(nn.Module):
    def __init__(self, dim, use_residual=True, init_a=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.ones(1) * init_a)
        self.y = nn.Parameter(torch.ones(1, 1, dim))
        self.b = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.use_residual = use_residual

    def forward(self, x):
        residual = x
        x = torch.tanh(self.a * self.norm(x)) * self.y + self.b
        return x + residual if self.use_residual else x


class FrequencyAwareAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.low_linear_real = nn.Linear(dim, dim)
        self.low_linear_imag = nn.Linear(dim, dim)
        self.high_linear_real = nn.Linear(dim, dim)
        self.high_linear_imag = nn.Linear(dim, dim)
        self.high_dynamic_tanh = DynamicTanhBlock(dim)
        self.attn = nn.MultiheadAttention(dim, 1, batch_first=True)

    @staticmethod
    def _upsample_to(x, target_len):
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=True)
        return x.transpose(1, 2)

    def forward(self, x):
        _, num_instances, _ = x.shape
        if num_instances < 4:
            raise ValueError("FC_MIL requires at least four instances for FFT bands")

        x_fft = torch.fft.fft(x, dim=1)
        cutoff = num_instances // 4
        low = x_fft[:, :cutoff]
        high = x_fft[:, -cutoff:]
        low = self.low_linear_real(low.real) + self.low_linear_imag(low.imag)
        high = self.high_linear_real(high.real) + self.high_linear_imag(high.imag)
        query = torch.sigmoid(low)
        key = self.high_dynamic_tanh(high)
        query = self._upsample_to(query, num_instances)
        key = self._upsample_to(key, num_instances)
        return self.attn(query, key, x)


class FC_MIL(nn.Module):
    """Frequency-aware causal MIL.

    ``max_instances`` is an optional memory guard for very large bags.
    Random training indices are sorted before the FFT; evaluation uses evenly
    spaced indices. Set it to ``None`` to run the unbounded official model.
    """

    def __init__(
        self,
        in_dim,
        num_classes,
        hidden_dim=512,
        dropout=0.25,
        max_instances=None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_instances = max_instances
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.spatial_attention = nn.MultiheadAttention(
            hidden_dim, 1, batch_first=True
        )
        self.frequency_attention = FrequencyAwareAttention(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

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
            ]
            indices = indices.sort().values
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
            raise ValueError("FC_MIL expects [N, D] or [B, N, D]")
        original_instances = x.shape[1]
        x, indices = self._select_instances(x, apply_instance_limit)

        h = self.feature(x)
        # The official model discards spatial weights. Avoid materializing them.
        h_spatial, _ = self.spatial_attention(
            h, h, h, need_weights=False
        )
        h_frequency, frequency_attention = self.frequency_attention(h)
        combined = h_spatial + h_frequency
        bag_feature = combined.mean(dim=1)
        logits = self.classifier(bag_feature)
        sampled_attention = frequency_attention.mean(dim=1)

        full_attention = sampled_attention.new_zeros(
            sampled_attention.shape[0], original_instances
        )
        full_attention.index_copy_(1, indices, sampled_attention)
        wsi_attention = full_attention.unsqueeze(-1)
        if input_was_2d:
            wsi_attention = wsi_attention.squeeze(0)

        result = {
            "logits": logits,
            "sampled_features": x,
            "sampled_attention": sampled_attention,
            "sampling_indices": indices,
        }
        if return_WSI_feature:
            result["WSI_feature"] = bag_feature
        if return_WSI_attn:
            result["WSI_attn"] = wsi_attention
        return result


def causal_mil_loss(
    model,
    original_logits,
    bag_features,
    attention_weights,
    topk_ratio=0.03,
    lam=0.5,
):
    """Official FC-MIL drop/replace causal regularizer (batch size one)."""

    if bag_features.shape[0] != 1:
        raise ValueError("FC-MIL causal loss requires batch_size=1")
    num_instances = bag_features.shape[1]
    topk = max(1, int(num_instances * topk_ratio))
    with torch.no_grad():
        attention = attention_weights.reshape(-1)
        top_indices = torch.topk(attention, topk).indices
        low_indices = torch.topk(attention, topk, largest=False).indices
        original_probs = F.softmax(original_logits, dim=1).detach()

    dropped = bag_features.clone()
    dropped[:, top_indices] = 0
    dropped_logits = model(
        dropped, apply_instance_limit=False
    )["logits"]

    replaced = bag_features.clone()
    replaced[:, low_indices] = bag_features[:, top_indices]
    replaced_logits = model(
        replaced, apply_instance_limit=False
    )["logits"]

    drop_kl = F.kl_div(
        F.log_softmax(dropped_logits, dim=1),
        original_probs,
        reduction="batchmean",
    )
    replace_kl = F.kl_div(
        F.log_softmax(replaced_logits, dim=1),
        original_probs,
        reduction="batchmean",
    )
    return -lam * drop_kl + (1.0 - lam) * replace_kl
