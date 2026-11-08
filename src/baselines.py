"""Líneas base sin aprendizaje profundo (referencias inferiores).

B0 - Media: predice el conteo medio del conjunto etiquetado. Define R2 ~ 0: cualquier
     modelo útil debe superarla.
B1 - Color (HSV + componentes conexas): sin entrenamiento ni anotación. Umbral de
     "rojez" calibrado en validación; el área media de una manzana se estima en
     validación para dividir manchas que contienen varias. Representa la alternativa
     de visión clásica y justifica (o no) el uso de IA.

Uso: python src/baselines.py
"""
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

from common import RES, load_npz, load_splits, metrics


def red_mask(img, thr):
    x = img.astype(np.float32) / 255
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    redness = r - np.maximum(g, b)
    return redness > thr


def count_blobs(img, thr, unit_area, min_area=12):
    m = ndimage.binary_opening(red_mask(img, thr), iterations=1)
    lab, n = ndimage.label(m)
    if n == 0:
        return 0.0
    areas = ndimage.sum(m, lab, range(1, n + 1))
    areas = areas[areas >= min_area]
    return float(np.maximum(1, np.round(areas / unit_area)).sum())


def main():
    sp = load_splits()
    out = {}
    tr_counts = [len(load_npz(s)[2]) for s in sp["train"]]
    mu = float(np.mean(tr_counts))
    gt_test = [len(load_npz(s)[2]) for s in sp["test"]]
    out["B0_media"] = {"test": metrics([mu] * len(gt_test), gt_test), "pred": mu}

    val = [(load_npz(s)[0], len(load_npz(s)[2])) for s in sp["val"]]
    best = None
    for thr in [0.06, 0.08, 0.10, 0.12, 0.15, 0.18]:
        for ua in [40, 60, 80, 110, 150]:
            m = metrics([count_blobs(i, thr, ua) for i, _ in val], [c for _, c in val])
            if best is None or m["MAE"] < best[0]["MAE"]:
                best = (m, thr, ua)
    _, thr, ua = best
    pred = [count_blobs(load_npz(s)[0], thr, ua) for s in sp["test"]]
    out["B1_color"] = {"test": metrics(pred, gt_test), "thr": thr, "unit_area": ua, "val": best[0],
                       "test_per_image": {s: (p, g) for s, p, g in zip(sp["test"], pred, gt_test)}}
    os.makedirs(os.path.join(RES, "runs"), exist_ok=True)
    json.dump(out, open(os.path.join(RES, "runs", "baselines.json"), "w"), indent=1)
    for k, v in out.items():
        print(k, v["test"])


if __name__ == "__main__":
    main()
