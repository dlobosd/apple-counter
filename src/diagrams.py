"""Diagramas del informe: arquitectura/flujo de datos y carta Gantt.  python src/diagrams.py"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from common import RES

INK, MUTED = "#1f2933", "#616e7c"
COL = {"datos": "#dbe9fb", "modelo": "#fde8d4", "eval": "#dff3e4", "app": "#efe3fa", "ext": "#eeeeee"}


def box(ax, x, y, w, h, title, sub, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=COL[kind], ec=INK, lw=1))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=8.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=6.6, color=MUTED, linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.08, label, ha="center", fontsize=6.3, color=MUTED)


def architecture():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    W, H = 1.75, 0.8
    # fila 1: datos
    box(ax, 0.1, 3.5, W, H, "MinneApple (HF)", "670 train + 331 test\nmáscaras COCO", "ext")
    box(ax, 2.2, 3.5, W, H, "download_minneapple", "descarga reanudable\nverifica tamaños", "datos")
    box(ax, 4.3, 3.5, W, H, "prepare_data", "centroides → densidad σ=4\nreescala 640 px, splits", "datos")
    box(ax, 6.4, 3.5, W, H, "data/processed", "index.csv · splits.json\n<img>.npz (img, pts, den)", "datos")
    arrow(ax, 1.85, 3.9, 2.2, 3.9)
    arrow(ax, 3.95, 3.9, 4.3, 3.9)
    arrow(ax, 6.05, 3.9, 6.4, 3.9)
    # fila 2: modelos
    box(ax, 0.1, 2.0, W, H, "Faster R-CNN COCO", "profesor zero-shot\n(torchvision, BSD-3)", "ext")
    box(ax, 2.2, 2.0, W, H, "pseudo_labels", "detecciones 'apple'\n→ pseudo-densidades", "modelo")
    box(ax, 4.3, 2.0, W, H, "train (U-Net)", "supervisada · preentren.\n+ ajuste fino; semillas", "modelo")
    box(ax, 6.4, 2.0, W, H, "CounTR (Colab)", "zero-shot / 3-shot\nnotebook → CSV", "ext")
    box(ax, 8.3, 2.0, 1.6, H, "baselines", "B0 media\nB1 color HSV", "modelo")
    arrow(ax, 1.85, 2.4, 2.2, 2.4)
    arrow(ax, 3.95, 2.4, 4.3, 2.4)
    arrow(ax, 7.2, 3.5, 5.3, 2.8, "npz")
    # fila 3: evaluación y app
    box(ax, 2.2, 0.4, W, H, "evaluate", "MAE · RMSE · R² · sesgo\nIC95 · análisis de fallos", "eval")
    box(ax, 4.3, 0.4, W, H, "results/", "runs/*.json · ckpt/*.pt\nsummary.json · figuras", "eval")
    box(ax, 6.4, 0.4, W, H, "MVP (Gradio)", "foto → conteo + mapa\n127.0.0.1, sin nube", "app")
    box(ax, 8.3, 0.4, 1.6, H, "Jefe técnico", "valida / corrige\nel conteo", "ext")
    arrow(ax, 5.2, 2.0, 5.2, 1.2, "checkpoints")
    arrow(ax, 7.2, 2.0, 4.0, 1.2)
    arrow(ax, 9.1, 2.0, 4.0, 1.0)
    arrow(ax, 3.95, 0.8, 4.3, 0.8)
    arrow(ax, 6.05, 0.8, 6.4, 0.8)
    arrow(ax, 8.15, 0.8, 8.3, 0.8)
    for k, lab in (("datos", "Datos"), ("modelo", "Modelos"), ("eval", "Evaluación"), ("app", "MVP"), ("ext", "Externo / usuario")):
        pass
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[k], ec=INK, lw=.6) for k in COL]
    ax.legend(handles, ["Datos", "Modelos", "Evaluación", "MVP", "Externo / usuario"], loc="lower left",
              ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.0, -0.06))
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_arquitectura.png"), dpi=170)
    plt.close(fig)


def gantt():
    # semanas desde el lunes 14-sep-2026 (semana 1) hasta la entrega final (semana 15, 21-dic-2026)
    tasks = [
        ("E1 · Formulación y estado del arte", 1, 1, "hecho"),
        ("Retroalimentación E1 y ajuste de alcance", 2, 2, "hecho"),
        ("Entorno, licencias y acceso a MinneApple", 4, 1, "hecho"),
        ("Pipeline de datos y particiones (Obj. 1)", 4, 2, "hecho"),
        ("Líneas base B0 / B1 (color, sin IA)", 5, 1, "hecho"),
        ("Pruebas de hardware: CPU · DirectML · CUDA", 5, 2, "hecho"),
        ("U-Net supervisada, línea base (Obj. 2)", 6, 1, "hecho"),
        ("Profesor detector + pseudo-densidades", 6, 2, "hecho"),
        ("Viabilidad CounTR: análisis + notebook (Obj. 3)", 7, 1, "hecho"),
        ("Grilla de eficiencia de etiquetas (Obj. 4)", 7, 3, "en curso"),
        ("MVP demostrable + informe E2", 8, 1, "hecho"),
        ("CounTR en Colab: inferencia 0/3-shot (Obj. 3)", 9, 1, "pendiente"),
        ("Análisis de fallos por escena (Obj. 5)", 10, 2, "pendiente"),
        ("Recomendación por presupuesto (Obj. 6)", 12, 1, "pendiente"),
        ("Informe final y presentación", 13, 3, "pendiente"),
    ]
    col = {"hecho": "#2a6fdb", "en curso": "#e07a1f", "pendiente": "#b8c2cc"}
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.axvspan(3.5, 8.5, color="#2a6fdb", alpha=0.06, lw=0)
    ax.text(6.0, -1.0, "período E2 · 5-oct a 7-nov", ha="center", color="#2a6fdb", fontsize=7.5)
    for i, (name, s, d, st) in enumerate(tasks):
        ax.barh(i, d, left=s - 0.5, color=col[st], height=0.6)
    ax.axvline(8.5, color="#c0392b", lw=1.2, ls="--")
    ax.text(8.55, -1.0, "E2 (7-nov)", color="#c0392b", fontsize=7.5)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks], fontsize=8)
    ax.invert_yaxis()
    ax.set_xticks(range(1, 16))
    import datetime as dt
    mon = dt.date(2026, 9, 14)
    ax.set_xticklabels([f"S{w}\n{(mon + dt.timedelta(weeks=w - 1)).strftime('%d/%m')}" for w in range(1, 16)], fontsize=7)
    ax.set_xlim(0.5, 15.5)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in col.values()]
    ax.legend(handles, list(col), loc="upper right", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_gantt.png"), dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    gantt()
