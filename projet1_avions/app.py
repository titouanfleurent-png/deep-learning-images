"""Démo : on met une photo d'avion, on obtient le constructeur, la famille, la variante et la carte Grad-CAM.
Lancement depuis projet1_avions/ : python app.py
Le modèle et les noms de classes sont dans models/ : pas besoin de télécharger le dataset.
"""
import json
from pathlib import Path

import gradio as gr
import torch

from src.dataset import LEVELS, build_transforms, crop_banner
from src.engine import AIRCRAFT_DIR, get_device, load_run, to_probs
from src.gradcam import explain

MODELS_DIR = Path(__file__).resolve().parent / "models"
RUN = "soft_efficientnet_b0_224_seed2"

meta = json.loads((MODELS_DIR / "classes.json").read_text())
classes = meta["classes"]
v2f, f2m = torch.tensor(meta["variant_to_family"]), torch.tensor(meta["family_to_manufacturer"])
device = get_device()
model, cfg = load_run(MODELS_DIR / RUN, classes, device)
FR = {"manufacturer": "Constructeur", "family": "Famille", "variant": "Variante"}

# Les exemples viennent de la validation et n'apparaissent que si le dataset est présent (photos non redistribuées)
EXAMPLES = [[str(p), True] for p in (AIRCRAFT_DIR / "images" / f"{i}.jpg" for i in ("2218512", "1468752", "1515540", "1716747"))
            if p.exists()] or None


def predict(image, has_banner):
    if image is None:
        return None, None, None, None
    image = image.convert("RGB")
    source = crop_banner(image) if has_banner else image
    x = build_transforms(False, cfg["img_size"])(source).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = to_probs(model(x), cfg["model"], v2f, f2m, classes)
    tops = [{classes[l][i]: float(p) for p, i in zip(*probs[l][0].topk(3))} for l in LEVELS]
    overlay, _ = explain(model, source, cfg["img_size"], banner=False)
    return (*tops, overlay)


demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil", label="Photo d'avion"),
            gr.Checkbox(value=False, label="Photo qui vient du dataset (enlève le bandeau de 20 px en bas)")],
    outputs=[gr.Label(label=FR[l]) for l in LEVELS] + [gr.Image(label="Où le modèle regarde (Grad-CAM)")],
    examples=EXAMPLES,
    title="Reconnaissance d'avions",
    description="Mets une photo d'avion : le modèle donne ses 3 meilleures réponses pour le constructeur, la famille et la variante. "
                "Il connaît 100 variantes (dataset FGVC-Aircraft).",
)

if __name__ == "__main__":
    demo.launch()
