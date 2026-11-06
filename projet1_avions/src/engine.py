"""Fonctions communes à l'entraînement, l'évaluation et la démo."""
from pathlib import Path

import torch

from .dataset import encode_labels, build_hierarchy, load_labels, LEVELS
from .metrics import lca_height_matrix, marginalize
from .model import AircraftClassifier

ROOT = Path(__file__).resolve().parents[1]
AIRCRAFT_DIR = ROOT / "data" / "fgvc-aircraft-2013b" / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_data():
    """Tableau encodé, noms de classes, tables hiérarchiques et matrice des hauteurs."""
    df, classes = encode_labels(load_labels(AIRCRAFT_DIR))
    v2f, f2m = build_hierarchy(df)
    return df, classes, v2f, f2m, lca_height_matrix(v2f, f2m)


def heads_for(kind, classes):
    """Le modèle hiérarchique a 3 têtes ; les modèles plat et soft n'en ont qu'une."""
    levels = LEVELS if kind == "hier" else ["variant"]
    return {level: len(classes[level]) for level in levels}


def load_run(run_dir, classes, device):
    """Recharge le meilleur modèle d'un run (choisi sur la validation)."""
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = AircraftClassifier(heads_for(cfg["model"], classes), cfg["backbone"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), cfg


def to_probs(logits, kind, v2f, f2m, classes):
    """Probabilités par niveau. Sans tête dédiée, famille et constructeur viennent des variantes."""
    probs = {name: torch.softmax(l, 1).cpu() for name, l in logits.items()}
    if kind != "hier":
        probs["family"], probs["manufacturer"] = marginalize(
            probs["variant"], v2f, f2m, len(classes["family"]), len(classes["manufacturer"]))
    return probs


@torch.no_grad()
def predict(model, loader, kind, device, v2f, f2m, classes):
    """Probabilités et labels sur tout un DataLoader (non mélangé)."""
    model.eval()
    probs, targets = {l: [] for l in LEVELS}, {l: [] for l in LEVELS}
    for images, labels in loader:
        batch = to_probs(model(images.to(device)), kind, v2f, f2m, classes)
        for level in LEVELS:
            probs[level].append(batch[level])
            targets[level].append(labels[level])
    return {l: torch.cat(probs[l]) for l in LEVELS}, {l: torch.cat(targets[l]) for l in LEVELS}
