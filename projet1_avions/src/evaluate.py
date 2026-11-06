"""Évaluation des runs terminés. Exemple : python -m src.evaluate --split test

Le split test ne doit être utilisé qu'une fois, quand tous les choix sont figés sur la validation.
Sorties dans results/ : <split>_metrics.csv (une ligne par run) et <split>_preds.npz (prédictions par image).
"""
import argparse

import numpy as np
import pandas as pd
import torch

from .dataset import LEVELS, make_loader
from .engine import AIRCRAFT_DIR, RESULTS_DIR, RUNS_DIR, get_device, load_data, load_run, predict
from .metrics import compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--device", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--pattern", default="*_seed*")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else get_device()
    df, classes, v2f, f2m, height = load_data()
    RESULTS_DIR.mkdir(exist_ok=True)
    rows, preds = [], {}
    for run_dir in sorted(RUNS_DIR.glob(args.pattern)):
        if not (run_dir / ".done").exists():
            continue
        model, cfg = load_run(run_dir, classes, device)
        loader = make_loader(df, AIRCRAFT_DIR, args.split, False, cfg["img_size"], 32, args.num_workers)
        probs, targets = predict(model, loader, cfg["model"], device, v2f, f2m, classes)
        m = compute_metrics(probs, targets, v2f, f2m, height)
        rows.append({"run": run_dir.name, "model": cfg["model"], "backbone": cfg["backbone"],
                     "img_size": cfg["img_size"], "seed": cfg["seed"], **m})
        for level in LEVELS:
            preds[f"{run_dir.name}/{level}"] = probs[level].argmax(1).numpy().astype(np.int16)
        preds[f"{run_dir.name}/top_variant_prob"] = probs["variant"].max(1).values.numpy().astype(np.float32)
        print(f"{run_dir.name} : variante {m['acc_variant']:.3f}, gravité {m['severity_mean_error']:.2f}", flush=True)
    for level in LEVELS:
        preds[f"target/{level}"] = targets[level].numpy().astype(np.int16)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{args.split}_metrics.csv", index=False)
    np.savez_compressed(RESULTS_DIR / f"{args.split}_preds.npz", **preds)


if __name__ == "__main__":
    main()
