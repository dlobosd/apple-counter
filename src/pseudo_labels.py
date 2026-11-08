"""Profesor detector -> pseudo-densidades (enfoque de Ramírez et al., 2026, adaptado).

El trabajo base usa YOLO como profesor. Aquí se usa Faster R-CNN de torchvision
preentrenado en COCO (clase 'apple'), por dos razones declaradas en el informe:
  1. Licencia: Ultralytics YOLO es AGPL-3.0; torchvision es BSD-3 (sin fricción para
     un prototipo que podría integrarse en un producto).
  2. El detector se usa tal cual (zero-shot sobre MinneApple), sin ver ninguna etiqueta
     del dataset: así el preentrenamiento no consume presupuesto de anotación.

Las detecciones con score >= umbral se convierten en puntos (centro de caja) y luego
en mapas de densidad con la misma gaussiana que las etiquetas reales.

Uso: python src/pseudo_labels.py --model mobilenet --thr 0.3
Salida: results/pseudo/<nombre>.npz (una densidad por imagen) + <nombre>_quality.json
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torchvision
from PIL import Image

from common import PROC, RES, load_splits, metrics
from prepare_data import RAW, density_map


def build(name):
    tv = torchvision.models.detection
    if name == "mobilenet":
        w = tv.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        m = tv.fasterrcnn_mobilenet_v3_large_fpn(weights=w, box_score_thresh=0.05)
    else:
        w = tv.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        m = tv.fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.05)
    return m.eval(), w.meta["categories"].index("apple")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenet", choices=["mobilenet", "resnet50"])
    ap.add_argument("--thr", type=float, nargs="+", default=[0.2, 0.3, 0.5])
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model, apple = build(a.model)
    sp = load_splits()
    stems = sp["train"] + sp["val"]
    os.makedirs(os.path.join(RES, "pseudo"), exist_ok=True)
    dets, t0 = {}, time.time()
    for i, s in enumerate(stems):
        img = Image.open(os.path.join(RAW, "train", "images", s + ".png")).convert("RGB")
        x = torchvision.transforms.functional.to_tensor(img)
        o = model([x])[0]
        k = o["labels"] == apple
        b, sc = o["boxes"][k].numpy(), o["scores"][k].numpy()
        dets[s] = (b, sc, img.size)
        if i % 50 == 0:
            print(f"{i}/{len(stems)} {time.time()-t0:.0f}s", flush=True)
    sec_per_img = (time.time() - t0) / len(stems)

    # calidad del profesor: se mide contra las etiquetas reales SOLO en validación
    from common import load_npz
    quality = {"model": a.model, "sec_per_img": sec_per_img, "by_thr": {}}
    for thr in a.thr:
        pred, gt = [], []
        for s in sp["val"]:
            b, sc, _ = dets[s]
            pred.append(int((sc >= thr).sum()))
            gt.append(len(load_npz(s)[2]))
        quality["by_thr"][str(thr)] = metrics(pred, gt)
        print(f"thr {thr}: {quality['by_thr'][str(thr)]}")
    best_thr = min(a.thr, key=lambda t: quality["by_thr"][str(t)]["MAE"])
    quality["chosen_thr"] = best_thr
    out = {}
    for s in sp["train"]:
        b, sc, (W, H) = dets[s]
        z = np.load(os.path.join(PROC, s + ".npz"))
        h, w = z["den"].shape
        f = w / W
        bb = b[sc >= best_thr]
        pts = np.stack([(bb[:, 0] + bb[:, 2]) / 2 * f, (bb[:, 1] + bb[:, 3]) / 2 * f], 1) if len(bb) else np.zeros((0, 2))
        out[s] = density_map(pts, h, w)
    name = f"frcnn_{a.model}"
    np.savez_compressed(os.path.join(RES, "pseudo", f"{name}.npz"), **out)
    json.dump(quality, open(os.path.join(RES, "pseudo", f"{name}_quality.json"), "w"), indent=1)
    print("umbral elegido", best_thr, f"{sec_per_img:.2f} s/img")


if __name__ == "__main__":
    main()
