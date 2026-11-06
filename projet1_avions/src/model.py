"""Modèles : un backbone CNN pré-entraîné ImageNet, surmonté d'une ou de 3 têtes de classification."""
import timm
import torch.nn as nn


class AircraftClassifier(nn.Module):
    """heads = {"variant": 100} pour le modèle plat,
    {"manufacturer": 30, "family": 70, "variant": 100} pour le modèle hiérarchique."""

    def __init__(self, heads: dict, backbone: str = "efficientnet_b0", pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)  # num_classes=0 : sans tête
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({name: nn.Linear(self.backbone.num_features, n) for name, n in heads.items()})

    def forward(self, x):
        features = self.dropout(self.backbone(x))
        return {name: head(features) for name, head in self.heads.items()}

    def freeze_backbone(self, frozen: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not frozen
