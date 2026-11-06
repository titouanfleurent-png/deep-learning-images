"""Analyse d'un split déjà évalué. Exemple : python -m src.analyze --split test

Sorties dans results/ : <split>_summary.csv, <split>_differences.csv, <split>_confusions.csv,
<split>_per_variant.csv, et les figures dans results/figures/.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .engine import RESULTS_DIR, RUNS_DIR, load_data
from .metrics import bootstrap_diff

MODELS = ["flat", "hier", "soft"]
NAMES = {"flat": "Plat", "hier": "Hiérarchique", "soft": "Soft labels"}
COLORS = {"flat": "#2a78d6", "hier": "#eb6834", "soft": "#1baf7a"}
SEEDS = [0, 1, 2]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLS = ["acc_variant", "balanced_acc_variant", "acc_family", "acc_manufacturer", "top5_variant", "severity_mean_error",
        "severity_mean_all", "errors_same_family", "errors_same_manufacturer", "errors_other_manufacturer",
        "grave_error_rate", "consistency"]


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def per_image(preds, height):
    """Par modèle, moyenne des 3 graines image par image : bonne réponse, gravité, erreur grave."""
    target = preds["target/variant"]
    out = {}
    for m in MODELS:
        h = np.stack([height[preds[f"{m}_efficientnet_b0_224_seed{s}/variant"], target] for s in SEEDS]).astype(float)
        out[m] = {"acc_variant": (h == 0).mean(0), "severity_mean_all": h.mean(0), "grave_error_rate": (h == 3).mean(0)}
    return out


def fig_levels(summary, path):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    levels = [("acc_manufacturer", "Constructeur (30)"), ("acc_family", "Famille (70)"), ("acc_variant", "Variante (100)")]
    for i, m in enumerate(MODELS):
        for j, (col, _) in enumerate(levels):
            v, s = summary.loc[m, (col, "mean")] * 100, summary.loc[m, (col, "std")] * 100
            ax.bar(j + (i - 1) * 0.26, v, 0.24, color=COLORS[m], label=NAMES[m] if j == 0 else None, yerr=s,
                   error_kw={"ecolor": INK, "lw": 1})
            ax.text(j + (i - 1) * 0.26, 3, f"{v:.1f}", ha="center", color="white", fontsize=9, fontweight="bold")
    ax.set_xticks(range(3), [l for _, l in levels], color=INK)
    ax.set_ylabel("Accuracy (%)", color=MUTED)
    ax.set_ylim(0, 100)
    style(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_severity(summary, path):
    """Répartition de toutes les images : bonne réponse, puis erreurs de plus en plus graves."""
    colors = ["#d9d8d3", "#f7c9b5", "#eb6834", "#a83a10"]
    labels = ["Bonne variante", "Même famille", "Même constructeur", "Autre constructeur"]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    for y, m in enumerate(MODELS[::-1]):
        acc = summary.loc[m, ("acc_variant", "mean")]
        parts = [acc] + [(1 - acc) * summary.loc[m, (c, "mean")] for c in
                         ("errors_same_family", "errors_same_manufacturer", "errors_other_manufacturer")]
        left = 0
        for k, p in enumerate(parts):
            ax.barh(y, p * 100, left=left, color=colors[k], edgecolor="white", linewidth=2, height=0.62,
                    label=labels[k] if y == 0 else None)
            ax.text(left + p * 50, y, f"{p * 100:.1f}", ha="center", va="center", fontsize=9,
                    color="white" if k >= 2 else INK)
            left += p * 100
    ax.set_yticks(range(3), [NAMES[m] for m in MODELS[::-1]], color=INK)
    ax.set_xlabel("Part des images (%)", color=MUTED)
    ax.set_xlim(0, 100)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(length=0, colors=MUTED)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.46, 1.2), fontsize=8.5)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_curves(path):
    """Accuracy variante sur la validation au fil des epochs, moyenne des 3 graines."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for m in MODELS:
        h = [json.loads((RUNS_DIR / f"{m}_efficientnet_b0_224_seed{s}" / "history.json").read_text()) for s in SEEDS]
        curve = np.mean([[e["val_acc_variant"] for e in run] for run in h], 0) * 100
        epochs = np.arange(1, len(curve) + 1)
        ax.plot(epochs, curve, color=COLORS[m], lw=2, label=NAMES[m])
    ax.axvline(3.5, color=MUTED, lw=1, ls="--")
    ax.text(3.7, 8, "Fin du gel du backbone", color=MUTED, fontsize=9)
    ax.set_xlabel("Epoch", color=MUTED)
    ax.set_ylabel("Accuracy variante, validation (%)", color=MUTED)
    ax.set_ylim(0, 80)
    style(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val", choices=["val", "test"])
    args = p.parse_args()
    split = args.split
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)

    _, classes, _, _, height = load_data()
    height = height.numpy()
    metrics = pd.read_csv(RESULTS_DIR / f"{split}_metrics.csv")
    metrics = metrics[metrics.model.isin(MODELS)]
    summary = metrics.groupby("model")[COLS].agg(["mean", "std"])
    summary.to_csv(RESULTS_DIR / f"{split}_summary.csv")

    preds = np.load(RESULTS_DIR / f"{split}_preds.npz")
    images = per_image(preds, height)
    rows = []
    for a, b in (("hier", "flat"), ("soft", "flat"), ("soft", "hier")):
        for metric in ("acc_variant", "severity_mean_all", "grave_error_rate"):
            mean, low, high = bootstrap_diff(images[a][metric], images[b][metric])
            rows.append({"comparison": f"{a} - {b}", "metric": metric, "diff": mean, "ci_low": low, "ci_high": high,
                         "significant": low > 0 or high < 0})
    diffs = pd.DataFrame(rows)
    diffs.to_csv(RESULTS_DIR / f"{split}_differences.csv", index=False)

    # Confusions et scores par variante : modèle soft, 3 graines cumulées
    target = preds["target/variant"]
    names = classes["variant"]
    conf = np.zeros((len(names), len(names)))
    for s in SEEDS:
        np.add.at(conf, (target, preds[f"soft_efficientnet_b0_224_seed{s}/variant"]), 1 / len(SEEDS))
    support = conf.sum(1)
    off = conf.copy()
    np.fill_diagonal(off, 0)
    top = np.argsort(off, axis=None)[::-1][:12]
    pairs = [(*np.unravel_index(i, off.shape),) for i in top]
    pd.DataFrame([{"vraie": names[t], "predite": names[q], "images": round(off[t, q], 1), "support": int(round(support[t])),
                   "hauteur": int(height[t, q])} for t, q in pairs]).to_csv(RESULTS_DIR / f"{split}_confusions.csv", index=False)
    per_variant = pd.DataFrame({"variante": names, "accuracy": np.diag(conf) / support, "support": support}).sort_values("accuracy")
    per_variant.to_csv(RESULTS_DIR / f"{split}_per_variant.csv", index=False)

    fig_levels(summary, fig_dir / f"{split}_niveaux.png")
    fig_severity(summary, fig_dir / f"{split}_gravite.png")
    fig_curves(fig_dir / "courbes_validation.png")

    off_diag = height[~np.eye(len(height), dtype=bool)]
    print("gravité d'une erreur tirée au hasard :", round(off_diag.mean(), 3))
    print(summary.xs("mean", axis=1, level=1).T.round(4))
    print(summary.xs("std", axis=1, level=1).T.round(4))
    print(diffs.round(4).to_string())
    print(pd.read_csv(RESULTS_DIR / f"{split}_confusions.csv").to_string())
    print(per_variant.head(6).round(3).to_string())
    print(per_variant.tail(6).round(3).to_string())


if __name__ == "__main__":
    main()
