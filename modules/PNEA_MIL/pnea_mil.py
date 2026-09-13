"""Positive and negative evidence aggregation MIL.

This is a clean implementation of the PNEA-MIL formulation published at
ACM BCB 2026. The reference implementation is available at
https://github.com/Chen-Jxiang/PNEA-MIL under GPL-3.0.
"""

import torch
import torch.nn as nn


class PNEA_MIL(nn.Module):
    """Aggregate class-specific non-negative evidence with max pooling."""

    def __init__(
        self,
        in_dim=512,
        evidence_dim=64,
        num_classes=2,
        lambda_l1=5.0,
    ):
        super().__init__()
        if evidence_dim < 1:
            raise ValueError("evidence_dim must be positive")
        self.in_dim = in_dim
        self.evidence_dim = evidence_dim
        self.num_classes = num_classes
        self.lambda_l1 = lambda_l1
        self.evidence_mapping = nn.Linear(
            in_dim, num_classes * evidence_dim
        )
        self.evidence_weight = nn.Parameter(
            0.1 * torch.abs(torch.rand(num_classes, evidence_dim))
        )
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(
        self,
        x,
        return_WSI_feature=False,
        return_WSI_attn=False,
        target_class=None,
        **_,
    ):
        input_was_2d = x.dim() == 2
        if input_was_2d:
            x = x.unsqueeze(0)
        if x.dim() != 3 or x.shape[-1] != self.in_dim:
            raise ValueError(
                f"PNEA_MIL expects [N, {self.in_dim}] or [B, N, {self.in_dim}]"
            )

        batch_size, num_instances, _ = x.shape
        patch_evidence = torch.relu(self.evidence_mapping(x)).view(
            batch_size,
            num_instances,
            self.num_classes,
            self.evidence_dim,
        )
        slide_evidence = patch_evidence.max(dim=1).values
        nonnegative_weight = torch.relu(self.evidence_weight)
        logits = torch.einsum(
            "bcd,cd->bc", slide_evidence, nonnegative_weight
        ) + self.bias
        sparsity_loss = self.lambda_l1 * nonnegative_weight.mean()

        if target_class is None:
            selected_class = logits.argmax(dim=-1)
        elif torch.is_tensor(target_class):
            selected_class = target_class.to(logits.device).long().reshape(-1)
            if selected_class.numel() == 1:
                selected_class = selected_class.expand(batch_size)
        else:
            selected_class = torch.full(
                (batch_size,),
                int(target_class),
                dtype=torch.long,
                device=logits.device,
            )
        if selected_class.shape != (batch_size,):
            raise ValueError("target_class must contain one class per bag")

        patch_class_evidence = torch.einsum(
            "bncd,cd->bnc", patch_evidence, nonnegative_weight
        )
        importance = patch_class_evidence.gather(
            2,
            selected_class[:, None, None].expand(-1, num_instances, 1),
        )
        bag_feature = slide_evidence.flatten(1)

        result = {
            "logits": logits,
            "patch_evidence": patch_evidence,
            "slide_evidence": slide_evidence,
            "nonnegative_weight": nonnegative_weight,
            "sparsity_loss": sparsity_loss,
            "target_class": selected_class,
        }
        if return_WSI_feature:
            result["WSI_feature"] = bag_feature
        if return_WSI_attn:
            # PNEA-MIL has evidence rather than attention. Return the selected
            # class's weighted patch evidence as instance importance.
            result["WSI_attn"] = importance.squeeze(0) if input_was_2d else importance
        return result
