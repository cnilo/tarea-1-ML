# Reproducción de PMR-GCN (Gao et al., IEEE T-ITS 2024) — Tarea 1 IPD440

Reproducción del experimento de predicción de flujo de pasajeros de metro del paper
**"Regularized Spatial-Temporal Graph Convolutional Networks for Metro Passenger Flow Prediction"**
(Gao, Liu, Huang, Wang, Li & Li, *IEEE Trans. Intelligent Transportation Systems* 25(9), 2024,
doi 10.1109/TITS.2024.3365179) usando el código oficial <https://github.com/cgao-comp/PMC-GCN>
y el único dataset incluido en ese repositorio (Hangzhou, `Dataset/HZ`).

## Estructura

```
README.md                       este archivo
requirements.txt
estado-del-arte-ml-metro.html   revisión bibliográfica que motivó la elección del paper
PMC-GCN/                        clon del repositorio oficial (no incluido; ver instalación)
repro/
  train_hz.py     entrenamiento + evaluación (réplica de New_train_batch_with_two_channel.py
                  y predict_batch_with_two_channel.py, configurado para HZ)
  baselines.py    predictores ingenuos sobre el mismo split/ventanas de test
  analyze.py      tablas y figuras a partir de runs/ y results/
  runs/<nombre>/  history.json (métricas por época), summary.json, best_net*.pt, test_predictions.npz
  results/        baselines.json, runs_summary.json, figuras .png
  logs/           salida estándar de cada corrida
  informe/        informe (no incluido en el repositorio; se entrega por Aula USM)
```

## Instalación y ejecución

```bash
git clone https://github.com/cgao-comp/PMC-GCN.git        # en la raíz del repo, junto a repro/
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python repro/baselines.py                                  # baselines (segundos)
python repro/train_hz.py --epochs 800 --seed 0             # run principal (config. del repo; ~3 h en M5 Pro/MPS)
python repro/train_hz.py --epochs 200 --seed 0 --single_channel   # ablación: sin 2ª red ni KL
python repro/train_hz.py --epochs 200 --seed 1             # variabilidad por semilla
python repro/train_hz.py --epochs 200 --seed 0 --loss mae  # pérdida MAE declarada en el paper (el código usa MSE)
python repro/analyze.py                                    # tablas + figuras en repro/results
python -c "import typst; typst.compile('repro/informe/informe.typ', output='repro/informe/informe.pdf', root='repro')"
```

Opciones útiles de `train_hz.py`: `--device {auto,cpu,mps,cuda}`, `--loss {mse,mae}` (mse = código oficial, mae = Ec. 16 del paper), `--kl_impl {log_target,original}`
(`original` reproduce el NaN del código oficial), `--drop_header_row` (descarta la fila de cabecera
que el código original lee como datos), `--eval_every N`.

## Diferencias respecto al código oficial (todas documentadas en el informe)

1. **Regularización KL**: `F.kl_div(log_softmax(p), softmax(q))` produce gradientes NaN en el primer
   paso con los datos HZ (el softmax sobre salidas en escala de pasajeros subdesborda a 0 exacto en ~74 %
   de las entradas). Se usa la forma equivalente `log_target=True`; el valor de la pérdida es idéntico.
2. Semilla fija, dispositivo configurable (el original asume `cuda:0` o CPU), sin `tensorboardX`.
3. Se registran métricas de test en cada época sólo para análisis; la selección del checkpoint se hace
   igual que en el repo (mínima pérdida de entrenamiento acumulada).
4. `time_num=69` y rutas de HZ: en el repo están comentadas (el script está configurado para SH, cuyo
   dataset no se distribuye).

## Hardware usado

MacBook Pro, Apple M5 Pro (15 núcleos: 5 rendimiento + 10 eficiencia), 24 GB RAM, macOS 26.6.2,
Python 3.14.3, PyTorch 2.14.0 (backend MPS). Sin GPU NVIDIA.
