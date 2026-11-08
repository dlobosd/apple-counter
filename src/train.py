"""Entrenamiento de la U-Net de densidad (Objetivos 2 y 4).

Ejemplos:
  # línea base supervisada con 100 % de las etiquetas
  python src/train.py --frac 1.0 --seed 0
  # eficiencia de etiquetas: 10 % de las imágenes de entrenamiento anotadas
  python src/train.py --frac 0.10 --seed 1
  # preentrenamiento guiado por detector (pseudo-densidades) y luego ajuste fino
  python src/train.py --frac 1.0 --seed 0 --pseudo results/pseudo/frcnn_pseudo.npz --tag pretrain_det
  python src/train.py --frac 0.10 --seed 1 --init results/ckpt/pretrain_det_s0.pt

Protocolo:
  * Subconjunto etiquetado = muestra aleatoria (semilla) de las imágenes de train.
  * Presupuesto fijo de iteraciones para todas las fracciones (no de épocas), para que
    la comparación mida la información disponible y no el cómputo.
  * Selección de modelo por MAE en validación cada `--eval-every` iteraciones.
  * El test solo se evalúa una vez, con el mejor checkpoint de validación.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import RES, DensityDataset, UNet, load_splits, metrics, predict_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--init", default=None, help="checkpoint para ajuste fino")
    ap.add_argument("--pseudo", default=None, help="npz con pseudo-densidades (entrena contra ellas)")
    ap.add_argument("--count-w", type=float, default=1e-2, help="peso del término L1 de conteo por recorte")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="cpu", help="cpu | cuda (GTX 1050 Ti) | dml (DirectML)")
    a = ap.parse_args()
    if a.device == "dml":
        import torch_directml
        dev = torch_directml.device()
    elif a.device == "cuda":
        dev = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        dev = torch.device("cpu")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    sp = load_splits()
    rng = np.random.default_rng(1000 + a.seed)
    train = sorted(sp["train"])
    k = max(2, int(round(a.frac * len(train))))
    labelled = sorted(rng.choice(train, k, replace=False).tolist())

    targets = None
    if a.pseudo:  # preentrenamiento: todas las imágenes de train, objetivo = pseudo-densidad
        z = np.load(a.pseudo)
        targets = {s: z[s] for s in train}
        labelled = train
    tag = a.tag or f"{'ft' if a.init else 'sup'}_f{a.frac:g}_s{a.seed}"
    os.makedirs(os.path.join(RES, "runs"), exist_ok=True)
    os.makedirs(os.path.join(RES, "ckpt"), exist_ok=True)

    model = UNet(a.base)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu"))
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.1)
    ds = DensityDataset(labelled, crop=a.crop, train=True, targets=targets)
    dl = DataLoader(ds, batch_size=min(a.bs, len(ds)), shuffle=True, drop_last=True, num_workers=0)

    curve, best, best_state, it, t0 = [], float("inf"), None, 0, time.time()
    while it < a.iters:
        for x, y, _ in dl:
            model.train()
            x, y = x.to(dev), y.to(dev)
            p = model(x)
            # MSE píxel a píxel + término de conteo por recorte (L1), ambos en escala DEN_SCALE
            loss = torch.mean((p - y) ** 2) + a.count_w * torch.mean(torch.abs(p.sum((1, 2, 3)) - y.sum((1, 2, 3))))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            it += 1
            if it % a.eval_every == 0 or it == a.iters:
                if a.pseudo:  # en preentrenamiento no se mira validación real: se guarda el último
                    val = {"MAE": float("nan")}
                    best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
                else:
                    pc = predict_counts(model, sp["val"], dev)
                    val = metrics([v[0] for v in pc.values()], [v[1] for v in pc.values()])
                    if val["MAE"] < best:
                        best = val["MAE"]
                        best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}
                curve.append({"iter": it, "loss": float(loss), "val_MAE": val["MAE"], "t": time.time() - t0})
                print(f"[{tag}] it {it} loss {float(loss):.4f} val MAE {val['MAE']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            if it >= a.iters:
                break

    model.load_state_dict({k_: v.to(dev) for k_, v in best_state.items()})
    torch.save(best_state, os.path.join(RES, "ckpt", f"{tag}.pt"))
    out = {"tag": tag, "args": vars(a), "n_labelled": len(labelled) if not a.pseudo else 0,
           "curve": curve, "train_seconds": time.time() - t0}
    if not a.pseudo:
        pc = predict_counts(model, sp["test"], dev)
        out["test"] = metrics([v[0] for v in pc.values()], [v[1] for v in pc.values()])
        out["test_per_image"] = pc
        out["val_best_MAE"] = best
        print(f"[{tag}] TEST {out['test']}")
    json.dump(out, open(os.path.join(RES, "runs", f"{tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
