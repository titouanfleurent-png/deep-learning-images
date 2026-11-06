"""Hiérarchie des classes et métriques : accuracy par niveau, gravité des erreurs, cohérence."""
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score


def lca_height_matrix(variant_to_family, family_to_manufacturer):
    """Matrice 100 x 100 : hauteur du plus proche ancêtre commun entre deux variantes.
    0 : même variante, 1 : même famille, 2 : même constructeur, 3 : constructeurs différents."""
    fam = variant_to_family
    man = family_to_manufacturer[fam]
    same_fam = fam[:, None] == fam[None, :]
    same_man = man[:, None] == man[None, :]
    height = torch.full((len(fam), len(fam)), 3)
    height[same_man] = 2
    height[same_fam] = 1
    height.fill_diagonal_(0)
    return height


def soft_targets(height, beta):
    """Soft labels de Bertinetto et al. (2020) : q(c | y) proportionnel à exp(-beta * d(c, y)),
    d étant la hauteur du plus proche ancêtre commun, normalisée entre 0 et 1."""
    return torch.softmax(-beta * height.float() / height.max(), dim=1)


def marginalize(variant_probs, variant_to_family, family_to_manufacturer, n_family, n_manufacturer):
    """Déduit les probabilités de famille et de constructeur en sommant celles des variantes."""
    fam_probs = torch.zeros(len(variant_probs), n_family).index_add_(1, variant_to_family, variant_probs)
    man_probs = torch.zeros(len(variant_probs), n_manufacturer).index_add_(1, family_to_manufacturer, fam_probs)
    return fam_probs, man_probs


def compute_metrics(probs, targets, variant_to_family, family_to_manufacturer, height):
    """probs et targets : dictionnaires par niveau (tenseurs CPU). Renvoie un dictionnaire de scores."""
    out = {}
    for level in ("manufacturer", "family", "variant"):
        pred = probs[level].argmax(1)
        out[f"acc_{level}"] = (pred == targets[level]).float().mean().item()
        out[f"balanced_acc_{level}"] = balanced_accuracy_score(targets[level].numpy(), pred.numpy())
    out["top5_variant"] = (probs["variant"].topk(5, 1).indices == targets["variant"][:, None]).any(1).float().mean().item()

    pred_v, true_v = probs["variant"].argmax(1), targets["variant"]
    h = height[pred_v, true_v].float()
    errors = h[h > 0]
    out["severity_mean_error"] = errors.mean().item()         # gravité moyenne d'une erreur
    out["severity_mean_all"] = h.mean().item()                # gravité moyenne, bonnes réponses comptées à 0
    for k, name in ((1, "same_family"), (2, "same_manufacturer"), (3, "other_manufacturer")):
        out[f"errors_{name}"] = (errors == k).float().mean().item()   # part des erreurs de chaque gravité
    out["grave_error_rate"] = (h == 3).float().mean().item()  # part de toutes les images

    fam_ok = variant_to_family[pred_v] == probs["family"].argmax(1)
    man_ok = family_to_manufacturer[probs["family"].argmax(1)] == probs["manufacturer"].argmax(1)
    out["consistency"] = (fam_ok & man_ok).float().mean().item()
    return out


def bootstrap_diff(values_a, values_b, n_boot=2000, seed=0):
    """Intervalle de confiance à 95 % de la moyenne de a - b, par bootstrap apparié sur les images."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    means = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
