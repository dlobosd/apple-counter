"""MVP: contador de manzanas para el jefe técnico de campo.

Interfaz local (Gradio) que recibe una foto de la hilera y devuelve:
  * conteo estimado por cada modelo entrenado disponible (U-Net con distintas
    cantidades de etiquetas, con y sin preentrenamiento guiado por detector),
  * mapa de densidad superpuesto (dónde está contando el modelo),
  * línea base de color (sin IA) como referencia,
  * si la imagen pertenece al test de MinneApple, el conteo real y el error.
El técnico valida o corrige el número: la herramienta asiste, no reemplaza (Entrega 1, §Responsabilidad).

Uso: python src/app.py   -> http://127.0.0.1:7860
"""
import glob
import json
import os

import gradio as gr
import numpy as np
import torch
from matplotlib import cm
from PIL import Image

from baselines import count_blobs
from common import DEN_SCALE, PROC, RES, UNet, load_splits, to_tensor
from prepare_data import LONG_SIDE

LABELS = {
    "sup_f1_s0": "U-Net supervisada · 100 % etiquetas",
    "sup_f0.1_s0": "U-Net supervisada · 10 % etiquetas",
    "sup_f0.05_s0": "U-Net supervisada · 5 % etiquetas",
    "ft_f0.1_s0": "U-Net + preentren. detector · 10 % etiquetas",
    "ft_f0.05_s0": "U-Net + preentren. detector · 5 % etiquetas",
}


def load_models():
    ms = {}
    for tag, name in LABELS.items():
        p = os.path.join(RES, "ckpt", f"{tag}.pt")
        if os.path.exists(p):
            m = UNet(16)
            m.load_state_dict(torch.load(p, map_location="cpu"))
            ms[name] = m.eval()
    return ms


MODELS = load_models()
BASE = json.load(open(os.path.join(RES, "runs", "baselines.json")))["B1_color"]
SPLITS = load_splits()
TEST = sorted(SPLITS["test"])
OOD = sorted(SPLITS.get("ood", []))


def overlay(img, den):
    d = np.clip(den / (np.percentile(den, 99.5) + 1e-8), 0, 1)
    heat = (cm.inferno(d)[..., :3] * 255).astype(np.float32)
    a = np.clip(d * 1.2, 0, 0.85)[..., None]
    return (img * (1 - a) + heat * a).astype(np.uint8)


@torch.no_grad()
def run(image, test_name):
    gt = None
    if test_name:
        z = np.load(os.path.join(PROC, test_name + ".npz"))
        img = z["img"]
        gt = int(len(z["pts"])) if test_name in TEST else None
    elif image is not None:
        pil = Image.fromarray(image).convert("RGB")
        s = LONG_SIDE / max(pil.size)
        img = np.asarray(pil.resize((round(pil.size[0] * s), round(pil.size[1] * s)), Image.BILINEAR))
    else:
        return None, "Sube una foto o elige una imagen de prueba."
    rows, first = [], None
    for name, m in MODELS.items():
        den = m(to_tensor(img)[None])[0, 0].numpy() / DEN_SCALE
        c = float(den.sum())
        first = den if first is None else first
        err = f"{c - gt:+.1f}" if gt is not None else "—"
        rows.append(f"| {name} | **{c:.1f}** | {err} |")
    cb = count_blobs(img, BASE["thr"], BASE["unit_area"])
    rows.append(f"| Línea base color (sin IA) | {cb:.0f} | {f'{cb - gt:+.0f}' if gt is not None else '—'} |")
    if gt is not None:
        head = f"**Conteo real (anotación MinneApple): {gt}**\n\n"
    elif test_name in OOD:
        head = "_Imagen de otro huerto (fuera de dominio): sin anotación confiable; el técnico valida el conteo._\n\n"
    else:
        head = ("_Foto subida: no tiene conteo real, por eso la columna Error aparece como «—». "
                "Si la foto es muy distinta a MinneApple (otra variedad, distancia o encuadre), "
                "el conteo no es confiable: el técnico debe validarlo._\n\n")
    table = head + "| Modelo | Conteo estimado | Error |\n|---|---|---|\n" + "\n".join(rows)
    table += "\n\n_El mapa muestra la densidad del primer modelo de la lista: más brillo = más manzanas por zona._"
    return overlay(img, first), table


with gr.Blocks(title="Conteo de manzanas · MVP") as demo:
    gr.Markdown("## Conteo de manzanas con anotaciones escasas · MVP (CINF200, Entrega 2)\n"
                "Sube una foto de la hilera o elige una imagen del conjunto de prueba de MinneApple.")
    with gr.Row():
        with gr.Column():
            inp = gr.Image(label="Foto de la hilera", type="numpy")
            sel = gr.Dropdown(choices=[""] + TEST + OOD, value="", label="…o imagen de prueba (dataset*: otro huerto, sin conteo real)")
            btn = gr.Button("Contar", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Mapa de densidad estimado")
            out_md = gr.Markdown()
    btn.click(run, [inp, sel], [out_img, out_md])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
