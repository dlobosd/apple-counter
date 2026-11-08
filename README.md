# Conteo de manzanas con anotaciones escasas · CINF200 Entrega 2

Diego Lobos Díaz · Prof. guía Billy Peralta · UNAB

Sistema reproducible que compara contadores por mapa de densidad (U-Net supervisada,
U-Net con preentrenamiento guiado por detector) y un contador genérico (CounTR) sobre
MinneApple, bajo distintos presupuestos de anotación, más un MVP para el jefe técnico de campo.

## Ver el MVP (2 minutos, sin GPU)

```bash
pip install -r requirements.txt
python src/app.py
```

Abrir http://127.0.0.1:7860, subir una foto de manzanos y pulsar **Contar**. La app muestra el
conteo de cada modelo, la línea base de color y el mapa de densidad (dónde cuenta el modelo).
Captura y descripción en la sección 6 del informe (figura 2).

## Estructura

```
src/
  download_minneapple.py   descarga reanudable del espejo COCO (Hugging Face)
  prepare_data.py          máscaras -> centroides -> densidad (σ=4 px, 640 px), splits por video
  baselines.py             B0 media, B1 color HSV (sin IA)
  pseudo_labels.py         Faster R-CNN COCO (zero-shot) -> pseudo-densidades
  train.py                 U-Net de densidad (supervisada / preentrenamiento / ajuste fino)
  run_grid.py              grilla fracciones × semillas, reanudable
  evaluate.py              tablas, IC95, figuras, análisis de fallos -> results/summary.json
  diagrams.py              arquitectura y Gantt
  app.py                   MVP (Gradio, http://127.0.0.1:7860)
notebooks/countr_minneapple_colab.ipynb   CounTR 0-shot / 3-shot (GPU NVIDIA)
informe/build_informe.js                  genera el informe .docx desde results/
results/                                  runs/*.json, ckpt/*.pt, figuras, summary.json
```

## Reproducir

```bash
py -3.11 -m venv .venv-cuda
.venv-cuda/Scripts/python -m pip install "torch==2.14.1+cu126" torchvision --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple
.venv-cuda/Scripts/python -m pip install numpy scipy pillow matplotlib pandas pycocotools gradio
python src/download_minneapple.py          # ~1,9 GB
python src/prepare_data.py                 # ~7 min
python src/baselines.py
python src/pseudo_labels.py --model mobilenet --thr 0.05 0.1 0.2 0.3 0.5
python src/run_grid.py --device cuda --iters 1200 --bs 8   # GTX 1050 Ti
python src/evaluate.py
python src/app.py
```

CounTR: abrir `notebooks/countr_minneapple_colab.ipynb` en Colab (GPU T4), ejecutar todo y
copiar `countr_test_predictions.csv` a `results/countr/`; `evaluate.py` lo incorpora solo.

## Protocolo

* Test = test oficial de MinneApple (331 imágenes, 12.285 manzanas), evaluado una sola vez por corrida.
* Validación = 2 videos completos de train (sin fuga entre fotogramas vecinos).
* 1.200 iteraciones fijas (lote 8, peso de conteo 1e-2) para toda fracción; preentrenamiento 800; mejor checkpoint por MAE de validación; 3 semillas planificadas (semilla 0 completada).
* Métricas: MAE, RMSE, R², sesgo; media ± IC95 %.

## Contenido del zip entregado

Incluye código (`src/`), notebook de CounTR (`notebooks/`), splits (`data/processed/`), resultados
por corrida (`results/runs/*.json`) y los pesos de los 5 modelos del MVP (`results/ckpt/`).
Las figuras y tablas están en el informe PDF. No incluye las imágenes de MinneApple (~2,4 GB) ni
las pseudo-etiquetas: se regeneran con `download_minneapple.py`, `prepare_data.py` y
`pseudo_labels.py`. Sin el dataset, el MVP (`python src/app.py`) funciona subiendo una foto; las
imágenes de prueba del menú requieren haber corrido `prepare_data.py`.
