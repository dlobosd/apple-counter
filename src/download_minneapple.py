"""Descarga MinneApple (espejo COCO en Hugging Face: lauesa1/minne-apple-segmentation).

Fuente original: Häni, Roy e Isler (2020), https://conservancy.umn.edu/handle/11299/206575
El espejo de HF conserva las 670 imágenes de entrenamiento y 331 de prueba con
máscaras de instancia en formato COCO (licencia CC BY-NC-SA 3.0, uso académico).

Uso:  python src/download_minneapple.py [--split train|test|all] [--workers 8]
Es reanudable: salta archivos ya descargados con el tamaño esperado.
"""
import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

REPO = "lauesa1/minne-apple-segmentation"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{{}}"
RAW = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{}}"
ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "minneapple")


def listing(path):
    with urllib.request.urlopen(API.format(path)) as r:
        return json.load(r)


def fetch(item):
    dst = os.path.join(ROOT, item["path"])
    if os.path.exists(dst) and os.path.getsize(dst) == item.get("size", -1):
        return "skip"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    urllib.request.urlretrieve(RAW.format(item["path"]), tmp)
    os.replace(tmp, dst)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["train", "test", "all"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    splits = ["train", "test"] if a.split == "all" else [a.split]
    for s in splits:
        items = [x for x in listing(f"{s}/images") if x["type"] == "file"]
        items += [x for x in listing(s) if x["type"] == "file"]
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(fetch, items))
        print(s, len(items), "archivos;", res.count("ok"), "nuevos,", res.count("skip"), "ya estaban")


if __name__ == "__main__":
    main()
