"""Vérifie que l'environnement des projets DL est prêt (packages + GPU Apple MPS)."""
import importlib
import platform
import sys

print(f"Python {platform.python_version()} ({sys.executable})")

PACKAGES = ["torch", "torchvision", "timm", "ultralytics", "supervision", "sklearn",
            "cv2", "umap", "pytorch_grad_cam", "gradio", "matplotlib", "pandas", "tensorboard"]
missing = []
for name in PACKAGES:
    try:
        mod = importlib.import_module(name)
        print(f"  OK   {name:<17} {getattr(mod, '__version__', '')}")
    except Exception as e:
        missing.append(name)
        print(f"  KO   {name:<17} {type(e).__name__}: {e}")

import torch
mps = torch.backends.mps.is_available()
print(f"\nMPS disponible : {mps}")
if mps:
    x = torch.randn(2048, 2048, device="mps")
    y = (x @ x).sum().item()
    print("Calcul test sur le GPU Apple : OK")

print("\nEnvironnement prêt." if not missing and mps else f"\nÀ corriger : {missing or 'MPS indisponible'}")
