"""Model creation helpers."""
from __future__ import annotations

from typing import Callable, Dict

import torch
import torchvision.models as models


_MODEL_BUILDERS: Dict[str, Callable[[bool], torch.nn.Module]] = {
    "vgg16": lambda pretrained: models.vgg16(weights=models.VGG16_Weights.DEFAULT if pretrained else None),
    "resnet50": lambda pretrained: models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None),
    "mobilenetv2": lambda pretrained: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None),
    "googlenet": lambda pretrained: models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT if pretrained else None),
}


def _replace_classifier(model: torch.nn.Module, model_name: str, num_classes: int) -> torch.nn.Module:
    if model_name == "vgg16":
        in_features = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "mobilenetv2":
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    elif model_name == "googlenet":
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def create_model(model_name: str, num_classes: int, *, pretrained: bool = True,
                 freeze_backbone: bool = False) -> torch.nn.Module:
    """Instantiate a torchvision model with a custom classifier head."""

    try:
        builder = _MODEL_BUILDERS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {model_name}") from exc

    model = builder(pretrained)
    model = _replace_classifier(model, model_name, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and name != "fc.weight" and name != "fc.bias":
                param.requires_grad = False

    return model


def build_feature_extractor(model: torch.nn.Module, model_name: str) -> torch.nn.Sequential:
    """Return the trunk of the network for embedding extraction."""

    if model_name == "vgg16":
        return torch.nn.Sequential(model.features, model.avgpool)
    if model_name == "resnet50":
        modules = list(model.children())[:-1]
        return torch.nn.Sequential(*modules)
    if model_name == "mobilenetv2":
        return torch.nn.Sequential(model.features, torch.nn.AdaptiveAvgPool2d((1, 1)))
    if model_name == "googlenet":
        modules = list(model.children())[:-2]
        return torch.nn.Sequential(*modules)
    raise ValueError(f"Unsupported model: {model_name}")


__all__ = ["create_model", "build_feature_extractor"]
