"""Agrega todas las corridas en tablas y figuras para el informe y el MVP.

  python src/evaluate.py
Salidas en results/: summary.json, tabla_resultados.csv, fig_*.png
"""
import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import DEN_SCALE, PROC, RES, UNet, load_npz, load_splits, metrics, to_tensor

C_SUP, C_DET, C_REF = "#2a6fdb", "#e07a1f", "#7a7a7a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return 1.96 * v.std(ddof=1) / np.sqrt(len(v))


def variedad(stem):
    # 19-sep-2015: variedad amarilla (ausente en train); 21-sep-2015: variedad roja
    return "amarilla" if stem.startswith("20150919") else "roja"


def por_variedad(per_image):
    out = {}
    for v in ("roja", "amarilla"):
        it = [(p, g) for s_, (p, g) in per_image.items() if variedad(s_) == v]
        if it:
            m = metrics([a for a, _ in it], [b for _, b in it])
            out[f"MAE_{v}"], out[f"bias_{v}"], out[f"n_{v}"] = m["MAE"], m["bias"], m["n"]
    return out


def load_runs():
    runs = {}
    for p in glob.glob(os.path.join(RES, "runs", "*.json")):
        r = json.load(open(p))
        if "test" in r and "tag" in r:
            runs[r["tag"]] = r
    return runs


def by_method(runs):
    g = defaultdict(lambda: defaultdict(list))
    for t, r in runs.items():
        meth = "det" if t.startswith("ft_") else "sup" if t.startswith("sup_") else None
        if meth:
            g[meth][r["args"]["frac"]].append(r)
    return g


def main():
    sp = load_splits()
    runs = load_runs()
    base = json.load(open(os.path.join(RES, "runs", "baselines.json")))
    n_train = len(sp["train"])
    g = by_method(runs)
    table = []
    for meth, fr in g.items():
        for f, rs in sorted(fr.items()):
            row = {"metodo": "U-Net supervisada" if meth == "sup" else "U-Net + preentren. detector",
                   "fraccion": f, "imagenes_etiquetadas": rs[0]["n_labelled"], "semillas": len(rs)}
            for m in ("MAE", "RMSE", "R2", "bias"):
                v = [r["test"][m] for r in rs]
                row[m] = float(np.mean(v))
                row[m + "_ic95"] = ci95(v)
            pv = [por_variedad(r["test_per_image"]) for r in rs]
            for k_ in pv[0]:
                row[k_] = float(np.mean([d[k_] for d in pv]))
            row["min_entrenamiento"] = float(np.mean([r["train_seconds"] for r in rs]) / 60)
            table.append(row)
    for k, name in (("B0_media", "Media del conjunto (B0)"), ("B1_color", "Color HSV, sin IA (B1)")):
        t = base[k]["test"]
        if k == "B1_color":
            pi = base[k]["test_per_image"]
        else:
            pi = {s_: (base[k]["pred"], float(len(load_npz(s_)[2]))) for s_ in sp["test"]}
        table.append({"metodo": name, "fraccion": "-", "imagenes_etiquetadas": "-", "semillas": 1,
                      **{m: t[m] for m in ("MAE", "RMSE", "R2", "bias")}, **por_variedad(pi)})
    countr = os.path.join(RES, "countr", "countr_test_predictions.csv")
    if os.path.exists(countr):
        import pandas as pd
        df = pd.read_csv(countr)
        df = df[df["stem"].isin(sp["test"])].copy()  # mismo test que la U-Net
        df["gt"] = [float(len(load_npz(s_)[2])) for s_ in df["stem"]]  # misma verdad de terreno
        for c in ("countr_0shot", "countr_3shot"):
            d = df.dropna(subset=[c])
            t = metrics(d[c], d["gt"])
            pi = {r.stem: (float(getattr(r, c)), float(r.gt)) for r in d.itertuples()}
            table.append({"metodo": f"CounTR {c.split('_')[1]}", "fraccion": 0, "imagenes_etiquetadas": 0,
                          "semillas": 1, **{m: t[m] for m in ("MAE", "RMSE", "R2", "bias")}, **por_variedad(pi)})
    with open(os.path.join(RES, "tabla_resultados.csv"), "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in table for k in r}, key=lambda k: list(table[0]).index(k) if k in table[0] else 99)
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(table)

    # ---- Figura 1: curva de eficiencia de etiquetas
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    b0row = next(r for r in table if r["metodo"].startswith("Media"))
    b1row = next(r for r in table if r["metodo"].startswith("Color"))
    for ax, v in zip(axes, ("roja", "amarilla")):
        for meth, col, lab in (("sup", C_SUP, "U-Net supervisada"), ("det", C_DET, "U-Net + preentren. detector")):
            if meth not in g:
                continue
            fr = sorted(g[meth])
            n = [g[meth][f][0]["n_labelled"] for f in fr]
            vals = [[por_variedad(r["test_per_image"])[f"MAE_{v}"] for r in g[meth][f]] for f in fr]
            ax.errorbar(n, [np.mean(x) for x in vals], yerr=[ci95(x) for x in vals], marker="o", color=col, label=lab, capsize=3, lw=2)
        ax.axhline(b0row[f"MAE_{v}"], color=C_REF, ls="--", lw=1, label="Media (B0)")
        ax.axhline(b1row[f"MAE_{v}"], color=C_REF, ls=":", lw=1, label="Color (B1)")
        ax.set_xscale("log")
        ax.set_xlabel("Imágenes de entrenamiento anotadas (escala log)")
        ax.set_ylabel("MAE (manzanas por imagen)")
        ax.set_title("Variedad roja (vista en entrenamiento)" if v == "roja" else "Variedad amarilla (no vista)", fontsize=9)
        ax.set_xticks(sorted({r["n_labelled"] for m_ in g.values() for rs in m_.values() for r in rs}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Error de conteo en test según presupuesto de anotación ({len(sp['test'])} imágenes; media ± IC95 % si hay varias semillas)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_eficiencia_etiquetas.png"), dpi=160)
    plt.close(fig)

    # ---- Figura 2: dispersión predicho vs real (mejor configuración con 100 % y 10 %)
    sel = [t for t in ("sup_f1_s0", "sup_f0.1_s0", "ft_f0.1_s0") if t in runs]
    fig, axes = plt.subplots(1, len(sel), figsize=(3.3 * len(sel), 3.3), squeeze=False)
    for ax, t in zip(axes[0], sel):
        pc = runs[t]["test_per_image"]
        gt = np.array([v[1] for v in pc.values()])
        pr = np.array([v[0] for v in pc.values()])
        col = ["#d4a017" if variedad(k) == "amarilla" else "#c0392b" for k in pc]
        ax.scatter(gt, pr, s=9, alpha=.7, c=col)
        lim = max(gt.max(), pr.max()) * 1.05
        ax.plot([0, lim], [0, lim], color=C_REF, lw=1)
        m = runs[t]["test"]
        ax.set_title(f"{t}\nMAE {m['MAE']:.1f} · R² {m['R2']:.2f}", fontsize=9)
        ax.set_xlabel("Conteo real")
        ax.set_ylabel("Conteo estimado")
    from matplotlib.lines import Line2D
    axes[0][0].legend([Line2D([], [], marker="o", ls="", color="#c0392b"), Line2D([], [], marker="o", ls="", color="#d4a017")],
                      ["variedad roja (vista)", "variedad amarilla (no vista)"], frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_dispersion.png"), dpi=160)
    plt.close(fig)

    # ---- Error por subconjunto de test (dataset/cara) y por densidad: análisis de fallos (Obj. 5)
    import pandas as pd
    idx = pd.read_csv(os.path.join(PROC, "index.csv")).set_index("stem")
    fail = {}
    if "sup_f1_s0" in runs:
        pc = runs["sup_f1_s0"]["test_per_image"]
        df = pd.DataFrame([{"stem": s, "pred": v[0], "gt": v[1]} for s, v in pc.items()]).set_index("stem").join(idx)
        df["err"] = df.pred - df["gt"]
        df["abs"] = df.err.abs()
        df["rel"] = df["abs"] / df["gt"].clip(lower=1)
        fail["por_grupo"] = df.groupby("group").agg(n=("gt", "size"), conteo=("gt", "mean"), MAE=("abs", "mean"),
                                                    sesgo=("err", "mean")).round(2).reset_index().to_dict("records")
        df["densidad_q"] = pd.qcut(df["gt"], 4, labels=["Q1 baja", "Q2", "Q3", "Q4 alta"])
        fail["por_densidad"] = df.groupby("densidad_q", observed=True).agg(
            rango=("gt", lambda x: f"{int(x.min())}-{int(x.max())}"), MAE=("abs", "mean"), sesgo=("err", "mean"),
            error_rel=("rel", "mean")).round(2).reset_index().astype({"densidad_q": str}).to_dict("records")
        df["brillo_q"] = pd.qcut(df.brightness, 3, labels=["oscura", "media", "clara"])
        fail["por_brillo"] = df.groupby("brillo_q", observed=True).agg(MAE=("abs", "mean"), sesgo=("err", "mean")).round(2).reset_index().astype({"brillo_q": str}).to_dict("records")
        fail["peores"] = df.sort_values("abs", ascending=False).head(6)[["gt", "pred", "group"]].round(1).reset_index().to_dict("records")
        fail["corr_err_brillo"] = float(np.corrcoef(df.err, df.brightness)[0, 1])
        fail["corr_err_conteo"] = float(np.corrcoef(df.err, df["gt"])[0, 1])

        # Figura 3: ejemplos cualitativos (mejor, mediano, peor) con mapa de densidad
        m = UNet(16)
        m.load_state_dict(torch.load(os.path.join(RES, "ckpt", "sup_f1_s0.pt"), map_location="cpu"))
        m.eval()
        order = df.sort_values("abs")
        picks = [order.index[0], order.index[len(order) // 2], fail["peores"][0]["stem"], fail["peores"][1]["stem"]]
        fig, axes = plt.subplots(2, 4, figsize=(10, 7.6))
        for j, s in enumerate(picks):
            img, den, pts = load_npz(s)
            with torch.no_grad():
                p = m(to_tensor(img)[None])[0, 0].numpy() / DEN_SCALE
            axes[0, j].imshow(img)
            axes[0, j].scatter(pts[:, 0], pts[:, 1], s=6, c="cyan", marker="+", lw=.7)
            axes[0, j].set_title(f"{s[:22]}\nreal {len(pts)}", fontsize=8)
            axes[1, j].imshow(img)
            axes[1, j].imshow(p, cmap="inferno", alpha=.6)
            axes[1, j].set_title(f"estimado {p.sum():.1f} (error {p.sum()-len(pts):+.1f})", fontsize=8)
            for a in axes[:, j]:
                a.axis("off")
        fig.suptitle("Arriba: anotación (+). Abajo: densidad estimada por U-Net (100 % etiquetas). Columnas: mejor, mediano, dos peores.", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, "fig_ejemplos.png"), dpi=130)
        plt.close(fig)

    # ---- Curvas de entrenamiento (val MAE vs iteración) para la fracción 100 % y 10 %
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for t, col in (("sup_f1_s0", C_SUP), ("sup_f0.1_s0", "#8fb3ef"), ("ft_f0.1_s0", C_DET)):
        if t in runs:
            c = runs[t]["curve"]
            ax.plot([x["iter"] for x in c], [x["val_MAE"] for x in c], marker=".", color=col, label=t)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("MAE validación")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig_curvas_entrenamiento.png"), dpi=160)
    plt.close(fig)

    pq = os.path.join(RES, "pseudo", "frcnn_mobilenet_quality.json")
    idx_all = pd.read_csv(os.path.join(PROC, "index.csv"))
    summary = {
        "splits": {s: len(sp[s]) for s in ("train", "val", "test")},
        "conteos": {s: {"total": int(idx_all[idx_all.split == s]["count"].sum()),
                        "media": float(idx_all[idx_all.split == s]["count"].mean()),
                        "min": int(idx_all[idx_all.split == s]["count"].min()),
                        "max": int(idx_all[idx_all.split == s]["count"].max())} for s in ("train", "val", "test")} | {"ood": {"n": int((idx_all.split == "ood").sum())}},
        "tabla": table, "fallos": fail, "baselines": {k: v["test"] for k, v in base.items()},
        "color_val": base["B1_color"]["val"],
        "profesor": json.load(open(pq)) if os.path.exists(pq) else None,
        "n_corridas": len(runs),
    }
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1, ensure_ascii=False)
    for r in table:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
