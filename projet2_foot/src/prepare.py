"""Fusionne gardien et joueur : 4 classes Roboflow -> 3 classes (ballon, joueur, arbitre).
Lancement depuis projet2_foot/ : python -m src.prepare
Les images ne sont pas copiées (liens physiques, aucune place en plus), seules les étiquettes sont réécrites.
Pas de lien symbolique : Ultralytics le suivrait et relirait les étiquettes d'origine à 4 classes.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "data"
SRC, DST = ROOT / "football_v12", ROOT / "football_3cls"
NAMES = ["ball", "player", "referee"]
REMAP = {0: 0, 1: 1, 2: 1, 3: 2}  # ball, goalkeeper -> player, player, referee


def main():
    for split in ("train", "valid", "test"):
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)
        images = DST / split / "images"
        if images.is_symlink():
            images.unlink()
        images.mkdir(exist_ok=True)
        for img in (SRC / split / "images").iterdir():
            if not (images / img.name).exists():
                os.link(img, images / img.name)
        for f in (SRC / split / "labels").glob("*.txt"):
            lines = []
            for line in f.read_text().splitlines():
                if line.strip():
                    cls, *box = line.split()
                    lines.append(" ".join([str(REMAP[int(cls)]), *box]))
            (DST / split / "labels" / f.name).write_text("\n".join(lines) + "\n")
    (DST / "data.yaml").write_text(
        f"path: {DST}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: {len(NAMES)}\nnames: {NAMES}\n")


if __name__ == "__main__":
    main()
