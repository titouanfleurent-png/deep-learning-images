"""Démo : déposer une photo d'avion, obtenir constructeur, famille, variante et carte Grad-CAM.
Lancement depuis projet1_avions/ : python app.py
"""
import gradio as gr
import torch

from src.dataset import LEVELS, build_transforms, crop_banner
from src.engine import AIRCRAFT_DIR, RUNS_DIR, get_device, load_data, load_run, to_probs
from src.gradcam import explain

RUN = "soft_efficientnet_b0_224_seed2"

device = get_device()
_, classes, v2f, f2m, _ = load_data()
model, cfg = load_run(RUNS_DIR / RUN, classes, device)
EXAMPLES = [[str(AIRCRAFT_DIR / "images" / f"{i}.jpg"), True] for i in ("2218512", "1468752", "1515540", "1716747")]  # validation
FR = {"manufacturer": "Constructeur", "family": "Famille", "variant": "Variante"}


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
            gr.Checkbox(value=False, label="Photo issue du dataset (retirer le bandeau de 20 px)")],
    outputs=[gr.Label(label=FR[l]) for l in LEVELS] + [gr.Image(label="Où le modèle regarde (Grad-CAM)")],
    examples=EXAMPLES,
    title="Reconnaissance d'avions",
    description="EfficientNet-B0 entraîné avec des soft labels hiérarchiques sur FGVC-Aircraft (30 constructeurs, 70 familles, 100 variantes).",
)

if __name__ == "__main__":
    demo.launch()
