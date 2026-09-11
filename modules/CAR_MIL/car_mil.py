"""Counterfactual-aware regularized attention MIL.

Adapted from the official CAR-MIL implementation:
https://github.com/ImaneCR/CAR-MIL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name):
    activations = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported CAR_MIL activation: {name}") from exc


class CAR_MIL(nn.Module):
    """Gated attention MIL with learned factual and counterfactual branches."""

    def __init__(
        self,
        in_dim=512,
        hidden_dim=512,
        attention_dim=128,
        num_classes=2,
        dropout=0.25,
        act="relu",
        attention_bias=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention_a = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim, bias=attention_bias),
            _activation(act),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim, bias=attention_bias),
            nn.Sigmoid(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.factual_attention = nn.Linear(
            attention_dim, 1, bias=attention_bias
        )
        self.counterfactual_attention = nn.Linear(
            attention_dim, 1, bias=attention_bias
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
                f"CAR_MIL expects [N, {self.in_dim}] or [B, N, {self.in_dim}]"
            )

        features = self.feature(x)
        gated = self.attention_a(features) * self.attention_b(features)
        factual_raw = self.factual_attention(gated).squeeze(-1)
        counterfactual_raw = self.counterfactual_attention(gated).squeeze(-1)
        factual_attention = F.softmax(factual_raw, dim=-1)
        counterfactual_attention = F.softmax(counterfactual_raw, dim=-1)
        factual_feature = torch.bmm(
            factual_attention.unsqueeze(1), features
        ).squeeze(1)
        counterfactual_feature = torch.bmm(
            counterfactual_attention.unsqueeze(1), features
        ).squeeze(1)
        logits = self.classifier(factual_feature)
        counterfactual_logits = self.classifier(counterfactual_feature)

        result = {
            "logits": logits,
            "counterfactual_logits": counterfactual_logits,
            "raw_attention": factual_raw,
            "counterfactual_raw_attention": counterfactual_raw,
            "attention": factual_attention,
            "counterfactual_attention": counterfactual_attention,
        }
        if return_WSI_feature:
            result["WSI_feature"] = factual_feature
        if return_WSI_attn:
            wsi_attention = factual_raw.unsqueeze(-1)
            result["WSI_attn"] = (
                wsi_attention.squeeze(0) if input_was_2d else wsi_attention
            )
        return result


def car_mil_loss(
    output,
    label,
    criterion,
    alpha_effect=0.2,
    alpha_attention=0.2,
    attention_loss="cosine",
):
    """Official CAR objective: factual, effect, and attention terms."""

    target = label
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        target = F.one_hot(
            label.long(), num_classes=output["logits"].shape[-1]
        ).float()
    main_loss = criterion(output["logits"], target)
    effect_loss = criterion(
        output["logits"] - output["counterfactual_logits"], target
    )
    factual = output["raw_attention"]
    counterfactual = output["counterfactual_raw_attention"]
    if attention_loss == "cosine":
        factual = F.normalize(factual, p=2, dim=-1)
        counterfactual = F.normalize(counterfactual, p=2, dim=-1)
        regularization = 1.0 - (factual * counterfactual).sum(dim=-1).mean()
    elif attention_loss == "l1":
        regularization = F.l1_loss(factual, counterfactual)
    else:
        raise ValueError("attention_loss must be 'cosine' or 'l1'")
    return (
        main_loss
        + alpha_effect * effect_loss
        + alpha_attention * regularization
    )
