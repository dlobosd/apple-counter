"""Corre la grilla de eficiencia de etiquetas (Objetivo 4), de forma reanudable.

  python src/run_grid.py --device dml --iters 1000

Orden: (1) supervisada en todas las fracciones y semillas, (2) preentrenamiento con
pseudo-densidades del detector, (3) ajuste fino desde ese preentrenamiento.
Salta las corridas cuyo results/runs/<tag>.json ya existe.
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def run(args, tag, py):
    if os.path.exists(os.path.join(RES, "runs", f"{tag}.json")):
        print("ya existe", tag)
        return
    log = open(os.path.join(RES, "logs", f"{tag}.log"), "w")
    print("corriendo", tag, flush=True)
    subprocess.run([py, "-W", "ignore", os.path.join(HERE, "train.py"), *args, "--tag", tag],
                   stdout=log, stderr=subprocess.STDOUT, check=True, cwd=HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--iters", type=int, default=1200)
    ap.add_argument("--fracs", type=float, nargs="+", default=[1.0, 0.25, 0.10, 0.05])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pseudo", default=os.path.join(RES, "pseudo", "frcnn_mobilenet.npz"))
    ap.add_argument("--only", choices=["sup", "det", "all"], default="all")
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(os.path.join(RES, "logs"), exist_ok=True)
    py = sys.executable
    common = ["--device", a.device, "--iters", str(a.iters), "--threads", str(a.threads),
              "--bs", str(a.bs), "--eval-every", str(a.eval_every)]
    if a.only in ("sup", "all"):
        for f in a.fracs:
            for s in a.seeds:
                run(common + ["--frac", str(f), "--seed", str(s)], f"sup_f{f:g}_s{s}", py)
    if a.only in ("det", "all"):
        pre = os.path.join(RES, "ckpt", "pretrain_det.pt")
        run(common + ["--pseudo", a.pseudo, "--seed", "0"], "pretrain_det", py)
        for f in a.fracs:
            if f >= 1.0:
                continue
            for s in a.seeds:
                run(common + ["--frac", str(f), "--seed", str(s), "--init", pre], f"ft_f{f:g}_s{s}", py)


if __name__ == "__main__":
    main()
