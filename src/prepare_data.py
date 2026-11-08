"""Pipeline de datos (Objetivo específico 1).

COCO (máscaras de instancia) -> centroides -> mapas de densidad gaussianos a
resolución de trabajo, + particiones sin fuga entre fotogramas vecinos.

Salidas en data/processed/:
  index.csv        una fila por imagen: split, grupo, conteo, densidad, estadísticas de escena
  <stem>.npz       imagen reescalada (uint8 HxWx3), puntos (N,2) y mapa de densidad (float32 HxW)
  splits.json      listas de imágenes por split + semilla + regla de agrupación

Decisiones (ver informe §3):
  * Resolución de trabajo: lado largo 640 px (original 720x1280, vertical). Las manzanas
    quedan con ~8-25 px de diámetro, suficiente para un mapa de densidad.
  * Gaussiana de sigma fijo = 4 px a la resolución de trabajo; la integral de cada
    gaussiana es 1, así que sum(densidad) == conteo exacto (se verifica abajo).
  * Fuga de datos: las imágenes de entrenamiento de MinneApple son fotogramas de
    video tomados en secuencia (nombre = fecha_hora_imageN). Dos fotogramas
    consecutivos muestran los mismos árboles, así que una partición aleatoria por
    imagen filtraría información. Se agrupa por secuencia de captura (mismo
    prefijo fecha_hora) y se parte por grupo.
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.ndimage import gaussian_filter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "minneapple")
OUT = os.path.join(ROOT, "data", "processed")
LONG_SIDE = 640
SIGMA = 4.0


def group_of(fname):
    # 20150919_174151_image1.png -> 20150919_174151 (un video = una hilera recorrida)
    m = re.match(r"(\d{8}_\d{6})_image\d+", fname)
    if m:
        return m.group(1)
    # dataset1_back_1.png -> dataset1_back
    m = re.match(r"(.+?)_(\d+)\.png$", fname)
    return m.group(1) if m else fname


def centroids(anns, h, w):
    pts = []
    for a in anns:
        seg = a["segmentation"]
        if isinstance(seg, list):
            rles = mask_utils.frPyObjects(seg, h, w)
            rle = mask_utils.merge(rles)
        elif isinstance(seg["counts"], list):
            rle = mask_utils.frPyObjects(seg, h, w)
        else:
            rle = seg
        m = mask_utils.decode(rle)
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            x, y, bw, bh = a["bbox"]
            pts.append((x + bw / 2, y + bh / 2, bw * bh))
        else:
            pts.append((xs.mean(), ys.mean(), len(xs)))
    return pts


def density_map(points, h, w, sigma=SIGMA):
    d = np.zeros((h, w), np.float32)
    for x, y in points:
        xi, yi = int(np.clip(round(x), 0, w - 1)), int(np.clip(round(y), 0, h - 1))
        d[yi, xi] += 1.0
    # mode="constant" conserva masa salvo en bordes; se renormaliza para que la suma sea exacta
    g = gaussian_filter(d, sigma, mode="constant")
    if len(points) and g.sum() > 0:
        g *= len(points) / g.sum()
    return g.astype(np.float32)


def process_split(split):
    coco = json.load(open(os.path.join(RAW, split, f"{split}.json")))
    by_img = defaultdict(list)
    for a in coco.get("annotations", []):
        by_img[a["image_id"]].append(a)
    rows = []
    for im in coco["images"]:
        fn = os.path.basename(im["file_name"])
        path = os.path.join(RAW, split, "images", fn)
        if not os.path.exists(path):
            continue
        pil = Image.open(path).convert("RGB")
        w, h = pil.size  # tamaño real del archivo (720x1280, vertical)
        pts = centroids(by_img[im["id"]], h, w)
        s = LONG_SIDE / max(h, w)
        nh, nw = round(h * s), round(w * s)
        img = np.asarray(pil.resize((nw, nh), Image.BILINEAR))
        P = np.array([(x * s, y * s) for x, y, _ in pts], np.float32).reshape(-1, 2)
        D = density_map(P, nh, nw)
        assert abs(D.sum() - len(P)) < 1e-2, (fn, D.sum(), len(P))
        stem = os.path.splitext(fn)[0]
        np.savez_compressed(os.path.join(OUT, f"{stem}.npz"), img=img, pts=P, den=D)
        areas = np.array([a for _, _, a in pts]) * s * s if pts else np.array([0.0])
        hsv = np.asarray(Image.fromarray(img).convert("HSV"), np.float32)
        rows.append({
            "stem": stem, "source": split, "group": group_of(fn), "count": len(P),
            "h": nh, "w": nw,
            "apple_area_med": float(np.median(areas)),
            "brightness": float(hsv[..., 2].mean() / 255),
            "density_per_mpx": len(P) / (nh * nw / 1e6),
        })
    return rows


def make_splits(rows, seed, val_frac=0.15, test_frac=0.20):
    """Partición por grupo sobre las imágenes anotadas. Si el split 'test' del espejo
    trae anotaciones, se usa como test oficial y solo se parte train en train/val."""
    rng = np.random.default_rng(seed)
    has_test = any(r["source"] == "test" and r["count"] > 0 for r in rows)
    pool = [r for r in rows if r["source"] == "train"]
    groups = sorted({r["group"] for r in pool})
    rng.shuffle(groups)
    n = len(pool)
    assign, acc = {}, 0
    tgt_test = 0 if has_test else test_frac * n
    cnt = defaultdict(int)
    for r in pool:
        cnt[r["group"]] += 1
    for g in groups:
        if acc < tgt_test:
            assign[g] = "test"
        elif acc < tgt_test + val_frac * n:
            assign[g] = "val"
        else:
            assign[g] = "train"
        acc += cnt[g]
    for r in rows:
        r["split"] = assign.get(r["group"], "test") if r["source"] == "train" else "test"
    return has_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = process_split("train")
    if os.path.exists(os.path.join(RAW, "test", "test.json")):
        rows += process_split("test")
    has_test = make_splits(rows, a.seed)
    with open(os.path.join(OUT, "index.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    splits = {s: [r["stem"] for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    json.dump({"seed": a.seed, "official_test": has_test, "long_side": LONG_SIDE, "sigma": SIGMA,
               "group_rule": "video de origen (prefijo fecha_hora) / prefijo de secuencia", **splits},
              open(os.path.join(OUT, "splits.json"), "w"), indent=1)
    for s, v in splits.items():
        c = [r["count"] for r in rows if r["split"] == s]
        print(f"{s}: {len(v)} imágenes, {len({r['group'] for r in rows if r['split']==s})} grupos, "
              f"conteo medio {np.mean(c):.1f} (min {min(c)}, max {max(c)})")


if __name__ == "__main__":
    main()
