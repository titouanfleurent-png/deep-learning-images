"""Planche Grad-CAM : images tirées au hasard sur la validation, sans sélection à la main.
Lancement : python -m src.figures_gradcam
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .engine import AIRCRAFT_DIR, RESULTS_DIR, RUNS_DIR, get_device, load_data, load_run
from .gradcam import explain

RUN = "soft_efficientnet_b0_224_seed2"
N = 4


def main():
    df, classes, _, _, height = load_data()
    df = df[df["split"] == "val"].reset_index(drop=True)
    preds = np.load(RESULTS_DIR / "val_preds.npz")
    pred, target = preds[f"{RUN}/variant"], preds["target/variant"]
    assert (df["variant_idx"].values == target).all()
    h = height.numpy()[pred, target]
    device = get_device()
    model, cfg = load_run(RUNS_DIR / RUN, classes, device)

    rng = np.random.default_rng(0)
    groups = [("Bonne variante", h == 0), ("Même famille", h == 1), ("Autre constructeur", h == 3)]
    fig, axes = plt.subplots(len(groups), N, figsize=(11, 8.2))
    for row, (label, mask) in enumerate(groups):
        for col, i in enumerate(rng.choice(np.flatnonzero(mask), N, replace=False)):
            image = Image.open(AIRCRAFT_DIR / "images" / f"{df.loc[i, 'image_id']}.jpg").convert("RGB")
            overlay, probs = explain(model, image, cfg["img_size"])
            ax = axes[row, col]
            ax.imshow(overlay)
            ax.set_xticks([]), ax.set_yticks([])
            ax.spines[:].set_visible(False)
            names = classes["variant"]
            ax.set_title(f"Vrai : {names[target[i]]}\nPrédit : {names[pred[i]]} ({probs.max():.0%})", fontsize=9)
            if col == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "gradcam_planche.png", dpi=170, bbox_inches="tight")


if __name__ == "__main__":
    main()
