"""nnMIL attention aggregator for MIL_BASELINE classification workflows."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NN_MIL(nn.Module):
    """Attention MIL with random feature subspaces and subspace ensembles.

    Attention is calculated in an ``hidden_dim``-wide feature subspace, while
    the weighted bag representation remains in the original embedding space.
    ``valid_mask`` prevents zero-padded patches from participating in the
    attention softmax.
    """

    def __init__(self, in_dim, hidden_dim=256, num_classes=2, dropout=0.25,
                 activation="softmax", feature_select=True,
                 eval_stride_divisor=4, cover_shuffle=True, cover_seed=42):
        super().__init__()
        if in_dim < 1 or hidden_dim < 1:
            raise ValueError("in_dim and hidden_dim must be positive")
        if eval_stride_divisor < 1:
            raise ValueError("eval_stride_divisor must be at least 1")
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.feature_select = bool(feature_select)
        self.eval_stride_divisor = int(eval_stride_divisor)
        self.cover_shuffle = bool(cover_shuffle)
        self.cover_seed = int(cover_seed)
        self.activation = activation
        self.V = nn.Linear(self.in_dim, self.hidden_dim)
        self.U = nn.Linear(self.in_dim, self.hidden_dim)
        self.w = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.classifier = nn.Linear(self.in_dim, self.num_classes)

    def _activate(self, scores):
        if self.activation != "softmax":
            raise ValueError("NN_MIL currently requires activation='softmax'")
        return F.softmax(scores, dim=1)

    @staticmethod
    def _linear_subspace(layer, x, indices):
        return F.linear(x.index_select(-1, indices), layer.weight.index_select(1, indices), layer.bias)

    def _attention(self, x, valid_mask=None, indices=None):
        if indices is None:
            a, b = torch.tanh(self.V(x)), torch.sigmoid(self.U(x))
        else:
            a = torch.tanh(self._linear_subspace(self.V, x, indices))
            b = torch.sigmoid(self._linear_subspace(self.U, x, indices))
        scores = self.w(self.dropout(a) * self.dropout(b))
        if valid_mask is not None:
            if valid_mask.shape != x.shape[:2]:
                raise ValueError("valid_mask must have shape [batch, patches]")
            scores = scores.masked_fill(~valid_mask.unsqueeze(-1), float("-inf"))
        return self._activate(scores)

    def _cover_indices(self, device):
        keep = min(self.hidden_dim, self.in_dim)
        if keep >= self.in_dim:
            return [torch.arange(self.in_dim, device=device)]
        if self.cover_shuffle:
            generator = torch.Generator(device=device)
            generator.manual_seed(self.cover_seed)
            permutation = torch.randperm(self.in_dim, generator=generator, device=device)
        else:
            permutation = torch.arange(self.in_dim, device=device)
        stride = max(1, keep // self.eval_stride_divisor)
        starts = list(range(0, self.in_dim - keep + 1, stride))
        if starts[-1] != self.in_dim - keep:
            starts.append(self.in_dim - keep)
        return [permutation[start:start + keep] for start in starts]

    @staticmethod
    def _entropy(probabilities):
        return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=-1)

    def forward(self, x, valid_mask=None, return_WSI_attn=False, return_WSI_feature=False):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected [batch, patches, {self.in_dim}], received {tuple(x.shape)}")
        if valid_mask is None:
            valid_mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=x.device, dtype=torch.bool)
        if not valid_mask.any(dim=1).all():
            raise ValueError("Each bag must contain at least one valid patch")

        keep = min(self.hidden_dim, self.in_dim)
        if self.training:
            indices = None
            if self.feature_select and keep < self.in_dim:
                indices = torch.randperm(self.in_dim, device=x.device)[:keep]
            attention = self._attention(x, valid_mask, indices)
            feature = torch.bmm(attention.transpose(1, 2), x).squeeze(1)
            result = {"logits": self.classifier(feature)}
            if return_WSI_attn:
                result["WSI_attn"] = attention.squeeze(-1)
            if return_WSI_feature:
                result["WSI_feature"] = feature
            return result

        chunks = [None] if not self.feature_select else self._cover_indices(x.device)
        logits_per_chunk, features_per_chunk, attentions_per_chunk = [], [], []
        for indices in chunks:
            attention = self._attention(x, valid_mask, indices)
            feature = torch.bmm(attention.transpose(1, 2), x).squeeze(1)
            logits_per_chunk.append(self.classifier(feature))
            features_per_chunk.append(feature)
            attentions_per_chunk.append(attention.squeeze(-1))
        chunk_logits = torch.stack(logits_per_chunk, dim=0)
        chunk_probabilities = F.softmax(chunk_logits, dim=-1)
        mean_probabilities = chunk_probabilities.mean(dim=0)
        total_entropy = self._entropy(mean_probabilities)
        entropy_each = self._entropy(chunk_probabilities)
        aleatoric_entropy = entropy_each.mean(dim=0)
        result = {
            "logits": chunk_logits.mean(dim=0),
            "chunk_logits": chunk_logits,
            "chunk_probabilities": chunk_probabilities,
            "total_entropy": total_entropy,
            "aleatoric_entropy": aleatoric_entropy,
            "mutual_information": (total_entropy - aleatoric_entropy).clamp_min(0),
            "probability_variance": chunk_probabilities.var(dim=0, unbiased=False),
        }
        if return_WSI_feature:
            result["WSI_feature"] = torch.stack(features_per_chunk, dim=0).mean(dim=0)
        if return_WSI_attn:
            result["WSI_attn"] = torch.stack(attentions_per_chunk, dim=0).mean(dim=0)
            result["WSI_attn_std"] = torch.stack(attentions_per_chunk, dim=0).std(dim=0, unbiased=False)
        return result
