"""Grad-CAM : où le modèle regarde-t-il pour reconnaître un avion ?"""
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from .dataset import build_transforms, crop_banner, denormalize


class VariantHead(torch.nn.Module):
    """Grad-CAM attend un modèle qui renvoie un tenseur : on isole la sortie « variante »."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["variant"]


def explain(model, pil_image, img_size=224, class_idx=None, banner=True):
    """Renvoie (image superposée à la carte de chaleur, probabilités des variantes)."""
    device = next(model.parameters()).device
    if banner:
        pil_image = crop_banner(pil_image)
    x = build_transforms(False, img_size)(pil_image).unsqueeze(0).to(device)
    wrapper = VariantHead(model).eval()
    probs = torch.softmax(wrapper(x), 1)[0].detach().cpu()
    target = int(probs.argmax()) if class_idx is None else class_idx
    with GradCAM(model=wrapper, target_layers=[model.backbone.blocks[-1]]) as cam:
        heatmap = cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]
    rgb = denormalize(x[0].cpu()).numpy().astype(np.float32)
    return show_cam_on_image(rgb, heatmap, use_rgb=True), probs
