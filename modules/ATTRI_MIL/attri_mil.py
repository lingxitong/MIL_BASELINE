"""AttriMIL aggregator and constraints.

Adapted from https://github.com/MedCAI/AttriMIL (Apache-2.0).
"""

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttention(nn.Module):
    def __init__(self, dim, attention_dim, dropout=0.0):
        super().__init__()
        branch_a = [nn.Linear(dim, attention_dim), nn.Tanh()]
        branch_b = [nn.Linear(dim, attention_dim), nn.Sigmoid()]
        if dropout > 0:
            branch_a.append(nn.Dropout(dropout))
            branch_b.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*branch_a)
        self.attention_b = nn.Sequential(*branch_b)
        self.attention_c = nn.Linear(attention_dim, 1)

    def forward(self, x):
        return self.attention_c(
            self.attention_a(x) * self.attention_b(x)
        )


class ATTRI_MIL(nn.Module):
    """Class-specific additive attribution MIL."""

    def __init__(
        self,
        in_dim,
        num_classes,
        attention_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        attention_dim = attention_dim or in_dim // 2
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.adaptor = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(),
            nn.Linear(in_dim // 2, in_dim),
        )
        self.attention_nets = nn.ModuleList(
            [
                GatedAttention(in_dim, attention_dim, dropout=dropout)
                for _ in range(num_classes)
            ]
        )
        self.classifiers = nn.ModuleList(
            [nn.Linear(in_dim, 1) for _ in range(num_classes)]
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
        if x.dim() != 3:
            raise ValueError("ATTRI_MIL expects [N, D] or [B, N, D]")

        h = x + self.adaptor(x)
        raw_attention = torch.stack(
            [network(h).squeeze(-1) for network in self.attention_nets],
            dim=1,
        )
        instance_scores = torch.stack(
            [classifier(h).squeeze(-1) for classifier in self.classifiers],
            dim=1,
        )
        exponential_attention = torch.exp(raw_attention)
        attribute_scores = instance_scores * exponential_attention
        logits = attribute_scores.sum(dim=-1) / exponential_attention.sum(
            dim=-1
        ) + self.bias
        predicted_class = logits.argmax(dim=1)

        if target_class is None:
            selected_class = predicted_class
        elif torch.is_tensor(target_class):
            selected_class = target_class.to(logits.device).long().reshape(-1)
            if selected_class.numel() == 1:
                selected_class = selected_class.expand(logits.shape[0])
        else:
            selected_class = torch.full(
                (logits.shape[0],),
                int(target_class),
                device=logits.device,
                dtype=torch.long,
            )
        if torch.any((selected_class < 0) | (selected_class >= self.num_classes)):
            raise ValueError("target_class is outside the configured classes")

        gather_index = selected_class[:, None, None].expand(
            -1, 1, h.shape[1]
        )
        selected_attribute = attribute_scores.gather(1, gather_index).transpose(
            1, 2
        )
        selected_raw = raw_attention.gather(1, gather_index).squeeze(1)
        selected_weights = F.softmax(selected_raw, dim=-1)
        bag_feature = torch.bmm(selected_weights.unsqueeze(1), h).squeeze(1)

        if input_was_2d:
            selected_attribute = selected_attribute.squeeze(0)
        result = {
            "logits": logits,
            "raw_attention": raw_attention,
            "instance_scores": instance_scores,
            "attribute_scores": attribute_scores,
            "predicted_class": predicted_class,
            "target_class": selected_class,
        }
        if return_WSI_feature:
            result["WSI_feature"] = bag_feature
        if return_WSI_attn:
            # AttriMIL has no single class-agnostic attention vector. The
            # requested/predicted class's signed attribute score is returned.
            result["WSI_attn"] = selected_attribute
        return result


def spatial_constraint(attribute_scores, nearest, num_classes):
    """Official local-prototype consistency loss with corrected indexing."""

    if nearest is None or nearest.numel() == 0:
        return attribute_scores.new_zeros(())
    if attribute_scores.shape[0] != 1:
        raise ValueError("AttriMIL spatial constraint requires batch_size=1")
    if nearest.dim() == 3:
        nearest = nearest.squeeze(0)
    nearest = nearest.to(attribute_scores.device).long()
    num_instances = attribute_scores.shape[-1]
    if nearest.dim() != 2 or nearest.shape[0] != num_instances:
        raise ValueError("nearest must have shape [num_instances, neighbors]")
    if nearest.min() < 0 or nearest.max() >= num_instances:
        raise ValueError("nearest contains an invalid instance index")

    loss = attribute_scores.new_zeros(())
    for class_index in range(1, num_classes):
        score = attribute_scores[0, class_index]
        neighbor_scores = score[nearest]
        prototype_indices = neighbor_scores.abs().argmax(dim=1, keepdim=True)
        local_prototype = neighbor_scores.gather(
            1, prototype_indices
        ).squeeze(1)
        loss = loss + torch.abs(torch.tanh(score - local_prototype)).mean()
    return loss


class AttributeMemory:
    """Per-class positive/negative FIFO queues used by the rank constraint."""

    def __init__(self, num_classes, queue_size=4):
        self.positive = [deque(maxlen=queue_size) for _ in range(num_classes)]
        self.negative = [deque(maxlen=queue_size) for _ in range(num_classes)]


def rank_constraint(
    data,
    label,
    model,
    attribute_scores,
    memory,
    num_classes,
):
    """Official inter-slide ranking loss, made graph- and device-safe."""

    if data.dim() == 3:
        if data.shape[0] != 1:
            raise ValueError("AttriMIL rank constraint requires batch_size=1")
        data = data.squeeze(0)
    label = int(label.reshape(-1)[0].item())
    loss = attribute_scores.new_zeros(())

    for class_index in range(num_classes):
        value, index = torch.topk(attribute_scores[0, class_index], k=1)
        top_feature = data[index.item() : index.item() + 1].detach()
        if label == class_index:
            memory.positive[class_index].append(top_feature)
            if not memory.negative[class_index]:
                continue
            queued_feature = memory.negative[class_index].popleft()
            memory.negative[class_index].append(queued_feature)
            queued_attribute = model(queued_feature)["attribute_scores"]
            queued_value = queued_attribute[0, class_index]
            if class_index != 0:
                loss = loss + torch.clamp(queued_value.mean() - value.mean(), min=0)
                loss = loss + torch.clamp(-value.mean(), min=0)
                loss = loss + torch.clamp(queued_value.mean(), min=0)
            else:
                loss = loss + torch.clamp(-value.mean(), min=0)
                loss = loss + torch.clamp(queued_value.mean(), min=0)
        else:
            memory.negative[class_index].append(top_feature)
            if not memory.positive[class_index]:
                continue
            queued_feature = memory.positive[class_index].popleft()
            memory.positive[class_index].append(queued_feature)
            queued_attribute = model(queued_feature)["attribute_scores"]
            queued_value = queued_attribute[0, class_index]
            if class_index != 0:
                loss = loss + torch.clamp(value.mean() - queued_value.mean(), min=0)
                loss = loss + torch.clamp(value.mean(), min=0)
            else:
                loss = loss + torch.clamp(value.mean(), min=0)
                loss = loss + torch.clamp(-queued_value.mean(), min=0)
    return loss / num_classes
