"""Re-partición tras detectar que las anotaciones del test del espejo están desalineadas.

Hallazgo (7-nov): en el split 'test' del espejo de Hugging Face los puntos anotados no
coinciden con las manzanas de la imagen (ver results/tuning/debug_align.png); en train sí.
Esto es coherente con que MinneApple nunca publicó el ground truth del test (se evaluaba en
un servidor). Decisión:
  * test etiquetado = videos completos de train, sorteados con semilla fija (sin elegir a mano),
    hasta cubrir >= 15 % de las imágenes; la validación existente (2 videos) no se toca;
  * las 331 imágenes del test oficial pasan a 'ood' (fuera de dominio, sin etiqueta usable):
    se usan solo cualitativamente en el MVP.
"""
import json
import os

import numpy as np
import pandas as pd

PROC = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def main(seed=2026, frac=0.15):
    idx = pd.read_csv(os.path.join(PROC, "index.csv"))
    sp = json.load(open(os.path.join(PROC, "splits.json")))
    val_groups = sorted(set(idx[idx.split == "val"].group))
    cand = sorted(set(idx[(idx.source == "train") & (~idx.group.isin(val_groups))].group))
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    n_tot = (idx.source == "train").sum()
    test_groups, acc = [], 0
    for g in cand:
        if acc >= frac * n_tot:
            break
        test_groups.append(g)
        acc += (idx.group == g).sum()
    idx.loc[idx.source == "test", "split"] = "ood"
    idx.loc[idx.group.isin(test_groups), "split"] = "test"
    idx.loc[(idx.source == "train") & ~idx.group.isin(test_groups + val_groups), "split"] = "train"
    idx.to_csv(os.path.join(PROC, "index.csv"), index=False)
    out = {k: v for k, v in sp.items() if k not in ("train", "val", "test")}
    out.update(official_test=False, test_rule=f"videos de train sorteados (semilla {seed}) hasta >= {frac:.0%}",
               test_groups=test_groups, val_groups=val_groups,
               nota="anotaciones del test oficial del espejo desalineadas -> split 'ood' (solo cualitativo)")
    for s in ("train", "val", "test", "ood"):
        out[s] = idx[idx.split == s].stem.tolist()
        print(s, len(out[s]), "imágenes,", int(idx[idx.split == s]["count"].sum()), "manzanas")
    print("test =", test_groups, "val =", val_groups)
    json.dump(out, open(os.path.join(PROC, "splits.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
