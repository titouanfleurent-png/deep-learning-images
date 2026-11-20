"""Exploration du dataset fusionné : nombre d'objets, tailles des boîtes, exemple annoté.
Lancement depuis projet2_foot/ : python -m src.explore
Les images Roboflow sont étirées de 1920 x 1080 à 1280 x 1280 : les tailles en pixels sont
recalculées dans l'image d'origine et dans l'image vue par YOLO (640 ou 1280 px).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "football_3cls"
FIG = ROOT / "results" / "figures"
NAMES = ["ball", "player", "referee"]
FR = {"ball": "Ballon", "player": "Joueur", "referee": "Arbitre"}
COLORS = {"ball": "#eb6834", "player": "#2a78d6", "referee": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_boxes():
    rows = []
    for split in ("train", "valid", "test"):
        for f in sorted((DATA / split / "labels").glob("*.txt")):
            for line in f.read_text().split("\n"):
                if line.strip():
                    c, x, y, w, h = line.split()
                    rows.append({"split": split, "image": f.stem, "cls": NAMES[int(c)],
                                 "x": float(x), "y": float(y), "w": float(w), "h": float(h)})
    df = pd.DataFrame(rows)
    df["w_orig"], df["h_orig"] = df.w * 1920, df.h * 1080          # pixels dans l'image d'origine
    for s in (640, 1280):
        df[f"side_{s}"] = np.sqrt(df.w * df.h) * s                  # côté moyen vu par YOLO à s px
    return df


def fig_sizes(df, path):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bins = np.logspace(np.log10(2), np.log10(300), 40)
    for c in NAMES:
        ax.hist(df[df.cls == c].side_640, bins=bins, color=COLORS[c], alpha=.85, label=FR[c],
                edgecolor="white", linewidth=.6)
    ax.set_xscale("log")
    ax.axvline(8, color=MUTED, lw=1, ls="--")
    ax.text(8.4, ax.get_ylim()[1] * .9, "8 px", color=MUTED, fontsize=9)
    ax.set_xlabel("Taille d'un objet dans une image de 640 px (côté moyen, pixels, échelle log)", color=MUTED)
    ax.set_ylabel("Nombre d'objets", color=MUTED)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, lw=.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def fig_example(df, path):
    """Une image d'entraînement remise au format 16:9, avec ses boîtes et un zoom sur le ballon."""
    img_name = df[(df.split == "train") & (df.cls == "ball")].image.iloc[0]
    img = Image.open(DATA / "train" / "images" / f"{img_name}.jpg").convert("RGB").resize((1920, 1080))
    boxes = df[(df.split == "train") & (df.image == img_name)]
    ball = boxes[boxes.cls == "ball"].iloc[0]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [3.2, 1]})
    for ax in (a, b):
        ax.imshow(img)
        ax.set_xticks([]), ax.set_yticks([])
        for _, r in boxes.iterrows():
            ax.add_patch(plt.Rectangle(((r.x - r.w / 2) * 1920, (r.y - r.h / 2) * 1080), r.w * 1920, r.h * 1080,
                                       fill=False, color=COLORS[r.cls], lw=1.2))
    cx, cy = ball.x * 1920, ball.y * 1080
    b.set_xlim(cx - 60, cx + 60), b.set_ylim(cy + 60, cy - 60)
    a.set_title("Une image du dataset et ses étiquettes", fontsize=10)
    b.set_title(f"Zoom sur le ballon ({ball.w_orig:.0f} x {ball.h_orig:.0f} px)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_boxes()
    counts = df.pivot_table(index="split", columns="cls", values="x", aggfunc="count").reindex(["train", "valid", "test"])
    counts["images"] = [df[df.split == s].image.nunique() for s in counts.index]
    print(counts)
    imgs = df.groupby(["split", "image"]).cls.apply(lambda s: (s == "ball").sum())
    print("images sans ballon :", int((imgs == 0).sum()), "sur", len(imgs), "| images avec plus d'un ballon :", int((imgs > 1).sum()))
    stats = df.groupby("cls")[["w_orig", "h_orig", "side_640", "side_1280"]].median().round(1)
    stats["part < 8 px à 640"] = df.groupby("cls").side_640.apply(lambda s: (s < 8).mean()).round(3)
    print(stats)
    print("objets par image (médiane) :", df.groupby("image").size().median())
    stats.to_csv(ROOT / "results" / "tailles_objets.csv")
    counts.to_csv(ROOT / "results" / "comptes.csv")
    fig_sizes(df, FIG / "tailles_objets.png")
    fig_example(df, FIG / "exemple_annote.png")


if __name__ == "__main__":
    main()
