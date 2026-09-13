"""AddMIL adapted from the authors' NeurIPS 2022 supplemental implementation.

Paper: https://proceedings.neurips.cc/paper/2022/hash/82764461a05e933cc2fd9d312e107d12-Abstract-Conference.html
Supplement: https://proceedings.neurips.cc/paper_files/paper/2022/file/82764461a05e933cc2fd9d312e107d12-Supplemental-Conference.zip
"""

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn


class StableSoftmax(nn.Module):
    """LogSoftmax followed by exp, as used in the AddMIL supplement."""

    def __init__(self, dim=0):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.log_softmax(inputs, dim=self.dim).exp()


def _build_mlp(
    input_dim,
    hidden_dims,
    output_dim,
    hidden_activation,
    use_batch_norm=False,
    track_bn_stats=True,
):
    dims = [input_dim, *hidden_dims, output_dim]
    layers = []
    for index, (in_features, out_features) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_features, out_features))
        if index < len(hidden_dims):
            if use_batch_norm:
                layers.append(
                    nn.BatchNorm1d(
                        out_features,
                        track_running_stats=track_bn_stats,
                    )
                )
            layers.append(copy.deepcopy(hidden_activation))
    return nn.Sequential(*layers)


class DefaultAttentionModule(nn.Module):
    """Attention pointer from the NeurIPS 2022 AddMIL supplement."""

    def __init__(
        self,
        input_dims,
        hidden_dims=(),
        hidden_activation=None,
        use_batch_norm=True,
        track_bn_stats=True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = tuple(hidden_dims)
        self.use_batch_norm = use_batch_norm
        self.track_bn_stats = track_bn_stats
        self.output_activation = StableSoftmax(dim=1)
        self.model = _build_mlp(
            input_dim=input_dims,
            hidden_dims=self.hidden_dims,
            output_dim=1,
            hidden_activation=hidden_activation or nn.LeakyReLU(0.2),
            use_batch_norm=use_batch_norm,
            track_bn_stats=track_bn_stats,
        )

    def forward(self, features, bag_size):
        if features.ndim != 2:
            raise ValueError(
                "DefaultAttentionModule expects flattened [B*N, D] features, "
                f"got {tuple(features.shape)}"
            )
        if bag_size <= 0 or features.shape[0] % bag_size != 0:
            raise ValueError(
                f"bag_size={bag_size} is incompatible with {features.shape[0]} instances"
            )

        attention_logits = self.model(features).reshape(-1, bag_size)
        return self.output_activation(attention_logits).unsqueeze(-1)


class AdditiveClassifier(nn.Module):
    """Class-wise instance predictor followed by additive bag aggregation."""

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_dims=(),
        hidden_activation=None,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = tuple(hidden_dims)
        self.model = _build_mlp(
            input_dim=input_dims,
            hidden_dims=self.hidden_dims,
            output_dim=output_dims,
            hidden_activation=hidden_activation or nn.ReLU(),
        )

    def forward(self, features, attention):
        if features.ndim != 3 or attention.shape != (*features.shape[:2], 1):
            raise ValueError(
                "features and attention must have shapes [B, N, D] and [B, N, 1], "
                f"got {tuple(features.shape)} and {tuple(attention.shape)}"
            )

        attended_features = attention * features
        patch_logits = self.model(attended_features)
        logits = patch_logits.sum(dim=1)
        return {
            "logits": logits,
            "patch_logits": patch_logits,
            "attended_features": attended_features,
        }


class ADD_MIL(nn.Module):
    """Additive MIL for pre-extracted pathology patch features.

    This is an adaptation of the authors' NeurIPS 2022 supplemental code. The
    pre-extracted patch embeddings are treated as ``f(x_i)`` in the paper:

        contribution_i = predictor(attention_i * f(x_i))
        bag_logits = sum_i contribution_i

    ``instance_contributions`` contains the exact signed, class-wise patch
    contributions. ``WSI_attn`` exposes one class for the generic MIL_BASELINE
    heatmap pipeline; it uses ``target_class`` when provided and otherwise the
    predicted class.
    """

    def __init__(
        self,
        L=512,
        D=128,
        num_classes=2,
        dropout=0,
        act=None,
        in_dim=512,
        rrt=None,
        *,
        hidden_dim=None,
        attention_hidden_dims=None,
        classifier_hidden_dims=None,
        use_batch_norm=True,
        track_bn_stats=True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = L
            if attention_hidden_dims is None:
                attention_hidden_dims = (L, D)
            if classifier_hidden_dims is None:
                classifier_hidden_dims = (L, D)
        if in_dim <= 0 or num_classes <= 0 or hidden_dim <= 0:
            raise ValueError("in_dim, num_classes, and hidden_dim must be positive")

        attention_hidden_dims = self._resolve_hidden_dims(
            attention_hidden_dims, hidden_dim
        )
        classifier_hidden_dims = self._resolve_hidden_dims(
            classifier_hidden_dims, hidden_dim
        )

        self.in_dim = in_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pointer = DefaultAttentionModule(
            input_dims=in_dim,
            hidden_dims=attention_hidden_dims,
            hidden_activation=nn.LeakyReLU(0.2),
            use_batch_norm=use_batch_norm,
            track_bn_stats=track_bn_stats,
        )
        self.classifier = AdditiveClassifier(
            input_dims=in_dim,
            output_dims=num_classes,
            hidden_dims=classifier_hidden_dims,
            hidden_activation=nn.ReLU(),
        )

    @staticmethod
    def _resolve_hidden_dims(hidden_dims, hidden_dim):
        if hidden_dims is None:
            return (hidden_dim, hidden_dim)
        if not isinstance(hidden_dims, Sequence) or isinstance(hidden_dims, (str, bytes)):
            raise TypeError("hidden dimensions must be a sequence of positive integers")
        hidden_dims = tuple(int(dim) for dim in hidden_dims)
        if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden dimensions must contain positive integers")
        return hidden_dims

    def _target_classes(self, logits, target_class):
        batch_size = logits.shape[0]
        if target_class is None:
            target_classes = logits.argmax(dim=-1)
        else:
            target_classes = torch.as_tensor(target_class, device=logits.device)
            if target_classes.ndim == 0:
                target_classes = target_classes.repeat(batch_size)
            target_classes = target_classes.reshape(-1).long()
            if target_classes.numel() != batch_size:
                raise ValueError(
                    "target_class must be a scalar or contain one class per bag"
                )
        if torch.any((target_classes < 0) | (target_classes >= self.num_classes)):
            raise ValueError(
                f"target_class must be between 0 and {self.num_classes - 1}"
            )
        return target_classes

    def forward(
        self,
        x,
        return_WSI_attn=False,
        return_WSI_feature=False,
        target_class=None,
    ):
        input_was_2d = x.ndim == 2
        if input_was_2d:
            x = x.unsqueeze(0)
        if x.ndim != 3:
            raise ValueError(f"expected [N, D] or [B, N, D], got {tuple(x.shape)}")

        batch_size, bag_size, feature_dim = x.shape
        if bag_size == 0:
            raise ValueError("AddMIL does not support empty bags")
        if feature_dim != self.in_dim:
            raise ValueError(
                f"expected feature dimension {self.in_dim}, got {feature_dim}"
            )

        flat_features = x.reshape(batch_size * bag_size, feature_dim)
        attention = self.pointer(flat_features, bag_size)
        classifier_output = self.classifier(x, attention)
        logits = classifier_output["logits"]
        contributions = classifier_output["patch_logits"]
        target_classes = self._target_classes(logits, target_class)
        gather_index = target_classes[:, None, None].expand(-1, bag_size, 1)
        selected_contributions = contributions.gather(dim=2, index=gather_index)

        forward_return = {
            "logits": logits,
            "patch_logits": contributions.squeeze(0) if input_was_2d else contributions,
            "instance_contributions": (
                contributions.squeeze(0) if input_was_2d else contributions
            ),
            "attention": attention.squeeze(0) if input_was_2d else attention,
            "predicted_class": logits.argmax(dim=-1),
            "target_class": target_classes,
        }
        if return_WSI_feature:
            forward_return["WSI_feature"] = classifier_output[
                "attended_features"
            ].sum(dim=1)
        if return_WSI_attn:
            forward_return["WSI_attn"] = (
                selected_contributions.squeeze(0)
                if input_was_2d
                else selected_contributions
            )
        return forward_return
