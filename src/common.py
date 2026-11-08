"""Utilidades compartidas: dataset, U-Net de densidad y métricas."""
import json
import os

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results")
DEN_SCALE = 100.0  # la densidad se multiplica por esta constante al entrenar (estabilidad numérica)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_splits():
    return json.load(open(os.path.join(PROC, "splits.json")))


def load_npz(stem):
    z = np.load(os.path.join(PROC, f"{stem}.npz"))
    return z["img"], z["den"], z["pts"]


def to_tensor(img):
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class DensityDataset(Dataset):
    """Recortes aleatorios + volteo horizontal para entrenar; imagen completa para evaluar.
    `targets` permite sustituir la densidad real por pseudo-densidades (preentrenamiento)."""

    def __init__(self, stems, crop=256, train=True, targets=None, cache=True, smin=0.75, smax=1.6):
        self.stems, self.crop, self.train, self.targets = list(stems), crop, train, targets
        self.smin, self.smax = smin, smax
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.stems)

    def _get(self, s):
        if self._cache is not None and s in self._cache:
            return self._cache[s]
        img, den, _ = load_npz(s)
        if self.targets is not None:
            den = self.targets[s]
        if self._cache is not None:
            self._cache[s] = (img, den)
        return img, den

    def __getitem__(self, i):
        s = self.stems[i]
        img, den = self._get(s)
        if self.train:
            h, w = den.shape
            c = self.crop
            # aumento de escala: se recorta una ventana de c/s px y se lleva a c px, de modo que
            # las manzanas aparecen s veces más grandes (s en [smin, smax]); la densidad se
            # reescala conservando su masa (= conteo de la ventana)
            s = np.exp(np.random.uniform(np.log(self.smin), np.log(self.smax))) if self.smax > 1 else 1.0
            cw = int(min(round(c / s), h, w))
            y = np.random.randint(0, h - cw + 1)
            x = np.random.randint(0, w - cw + 1)
            img, den = img[y:y + cw, x:x + cw], den[y:y + cw, x:x + cw]
            if cw != c:
                mass = float(den.sum())
                img = np.asarray(Image.fromarray(img).resize((c, c), Image.BILINEAR))
                den = np.asarray(Image.fromarray(den.astype(np.float32)).resize((c, c), Image.BILINEAR), np.float32)
                if den.sum() > 0:
                    den = den * (mass / den.sum())
            if np.random.rand() < 0.5:
                img, den = img[:, ::-1], den[:, ::-1]
        return to_tensor(img), torch.from_numpy(den.copy())[None] * DEN_SCALE, s


def _block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) de 4 niveles con salida de densidad no negativa.
    base=16 -> ~1.9 M parámetros: cabe en CPU y en la GPU local de 8 GB."""

    def __init__(self, base=16):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.e = nn.ModuleList([_block(3, c[0])] + [_block(c[k], c[k + 1]) for k in range(4)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(c[k + 1], c[k], 2, stride=2) for k in range(4)])
        self.d = nn.ModuleList([_block(c[k] * 2, c[k]) for k in range(4)])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        ph, pw = (-h) % 16, (-w) % 16
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        skips = []
        for k, blk in enumerate(self.e):
            x = blk(x if k == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for k in reversed(range(4)):
            x = self.d[k](torch.cat([self.up[k](x), skips[k]], 1))
        return F.relu(self.head(x))[..., :h, :w]


def metrics(pred, gt):
    pred, gt = np.asarray(pred, float), np.asarray(gt, float)
    err = pred - gt
    ss_res = (err ** 2).sum()
    ss_tot = ((gt - gt.mean()) ** 2).sum()
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(err.mean()),
        "n": int(len(gt)),
    }


@torch.no_grad()
def predict_counts(model, stems, device="cpu"):
    model.eval()
    out = {}
    for s in stems:
        img, den, _ = load_npz(s)
        p = model(to_tensor(img)[None].to(device))[0, 0].cpu().numpy() / DEN_SCALE
        out[s] = (float(p.sum()), float(den.sum()))
    return out
