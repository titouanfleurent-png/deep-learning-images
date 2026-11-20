"""Entraînement YOLO11 sur les 3 classes. Exemple : python -m src.train --model n --imgsz 640
Le meilleur modèle est choisi sur la validation (fitness d'Ultralytics) ; le test reste pour la fin.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "football_3cls" / "data.yaml"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["n", "s"], default="n")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps")
    args = p.parse_args()

    name = f"yolo11{args.model}_{args.imgsz}_seed{args.seed}"
    model = YOLO(f"yolo11{args.model}.pt")  # pré-entraîné sur COCO
    model.train(data=str(DATA), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, patience=30,
                seed=args.seed, deterministic=True, device=args.device, workers=4,
                project=str(ROOT / "runs"), name=name, exist_ok=True, plots=True)
    (ROOT / "runs" / name / ".done").touch()


if __name__ == "__main__":
    main()
