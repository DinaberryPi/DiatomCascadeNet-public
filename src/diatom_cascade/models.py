"""Canonical model architectures used by training, evaluation, and prediction."""

import timm
import torch
from torch import nn

from .config.model_config import HEAD_SPECS


def _make_head(input_dim, output_dim, spec_name):
    hidden_dims, dropouts = HEAD_SPECS[spec_name]
    if len(dropouts) != len(hidden_dims) + 1:
        raise ValueError(f"Invalid head specification for {spec_name}")

    layers = []
    current_dim = input_dim
    for hidden_dim, dropout in zip(hidden_dims, dropouts):
        layers.extend((nn.Dropout(dropout), nn.Linear(current_dim, hidden_dim), nn.ReLU()))
        current_dim = hidden_dim
    layers.extend((nn.Dropout(dropouts[-1]), nn.Linear(current_dim, output_dim)))
    return nn.Sequential(*layers)


def _make_backbone(model_name, pretrained):
    if not isinstance(pretrained, bool):
        raise TypeError("pretrained must be a bool")
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=0,
        global_pool="avg",
    )


class FlatClassifier(nn.Module):
    """Shared flat classifier for F-C, F-G, and F-S."""

    def __init__(self, num_classes, model_name, pretrained):
        super().__init__()
        self.backbone = _make_backbone(model_name, pretrained)
        self.classifier = _make_head(self.backbone.num_features, num_classes, "flat")

    def forward(self, x):
        return self.classifier(self.backbone(x))


class _HierarchicalClassifier(nn.Module):
    """Shared top-down architecture; public subclasses declare active levels only."""

    def __init__(self, levels, model_name, pretrained):
        super().__init__()
        self.backbone = _make_backbone(model_name, pretrained)
        feature_dim = self.backbone.num_features
        previous_outputs = 0
        for level_name, output_dim in levels:
            head = _make_head(
                feature_dim + previous_outputs,
                output_dim,
                level_name,
            )
            setattr(self, f"{level_name}_classifier", head)
            previous_outputs += output_dim
        self._level_names = tuple(level_name for level_name, _ in levels)

    def forward(self, x):
        features = self.backbone(x)
        logits = []
        probabilities = []
        inputs = [features]
        for level_name in self._level_names:
            head = getattr(self, f"{level_name}_classifier")
            level_logits = head(torch.cat(inputs, dim=1) if len(inputs) > 1 else features)
            logits.append(level_logits)
            level_probabilities = torch.softmax(level_logits, dim=1)
            probabilities.append(level_probabilities)
            inputs.append(level_probabilities)
        return tuple(logits + probabilities[:-1])


class ClassToOrderModel(_HierarchicalClassifier):
    """Canonical H-CO model."""

    def __init__(self, num_classes, num_orders, model_name, pretrained):
        super().__init__(
            (("class", num_classes), ("order", num_orders)),
            model_name,
            pretrained,
        )


class ThreeLevelHierarchicalModel(_HierarchicalClassifier):
    """Canonical H-COF model."""

    def __init__(self, num_classes, num_orders, num_families, model_name, pretrained):
        super().__init__(
            (
                ("class", num_classes),
                ("order", num_orders),
                ("family", num_families),
            ),
            model_name,
            pretrained,
        )


class HCOFGModel(_HierarchicalClassifier):
    """Canonical H-COFG model."""

    def __init__(
        self,
        num_classes,
        num_orders,
        num_families,
        num_genera,
        model_name,
        pretrained,
    ):
        super().__init__(
            (
                ("class", num_classes),
                ("order", num_orders),
                ("family", num_families),
                ("genus", num_genera),
            ),
            model_name,
            pretrained,
        )


class HCOFGSModel(_HierarchicalClassifier):
    """Canonical H-COFGS model."""

    def __init__(
        self,
        num_classes,
        num_orders,
        num_families,
        num_genera,
        num_species,
        model_name,
        pretrained,
    ):
        super().__init__(
            (
                ("class", num_classes),
                ("order", num_orders),
                ("family", num_families),
                ("genus", num_genera),
                ("species", num_species),
            ),
            model_name,
            pretrained,
        )

