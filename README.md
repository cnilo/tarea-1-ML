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

---

# Tarea 2 · Referencia estacional por tipo de día (`t2/`)

**Pregunta.** ¿Entregar a PMR-GCN la referencia estacional por tipo de día (el flujo del mismo intervalo en el
último día comparable: laboral, sábado, domingo o festivo), como canal de entrada y como base de una salida
residual, reduce su MAE de predicción por estación a 15–60 min en al menos un 10 %, y lo deja por debajo del
predictor ingenuo que usa esa misma referencia?

El destino de la investigación es **Metro de Santiago**. Mientras no hay datos de Santiago, el estudio se corre
como piloto en el metro de Hangzhou (el único dataset que distribuye el código oficial). El código no tiene nada
fijo de Hangzhou: todo lo específico de un metro vive en `t2/datasets/<nombre>.json`.

## Archivos

```
t2/
  datasets/
    hangzhou.json              configuración del piloto
    santiago.plantilla.json    plantilla para Metro de Santiago (copiar como santiago.json)
  common.py      clase Dataset: lectura, calendario, tipos de día, feriados, partición, ventanas, métricas
  linear.py      predictores ingenuos y regresión ridge por estación (con y sin referencia)
  train_gcn.py   PMR-GCN oficial, variantes base y ref (2 canales + salida residual)
  analyze.py     tabla principal, intervalos de confianza, figuras y macros LaTeX
  run_all.sh     corre todo para un dataset
  runs/<dataset>/      predicciones y métricas por corrida (los pesos .pt no se versionan)
  results/<dataset>/   summary.json, figuras y numeros.tex
  logs/<dataset>/      salida de cada corrida
```

## Ejecución

Requiere el clon de `PMC-GCN/` en la raíz (ver arriba) y las dependencias de `requirements.txt`.

```bash
bash t2/run_all.sh hangzhou                            # todo: ~1 h 40 min en Apple M5 Pro (MPS)
# o por partes, desde t2/:
python linear.py    --dataset hangzhou                 # ingenuos + ridge (segundos)
python train_gcn.py --dataset hangzhou --variant base --seed 0   # PMR-GCN sin referencia (~33 min)
python train_gcn.py --dataset hangzhou --variant ref  --seed 0   # PMR-GCN con referencia (~33 min)
python analyze.py   --dataset hangzhou                 # resultados en t2/results/hangzhou
```

## Migrar a Metro de Santiago

1. **Datos de flujo** (`flujo.csv`): una fila por intervalo de 15 minutos, días completos y consecutivos en orden
   cronológico, y una columna por estación con el número de validaciones (entradas) en ese intervalo. Primera
   fila con los nombres de estación. Se usa una grilla fija para todos los días (por ejemplo 05:30 a 23:30 = 72
   intervalos); los intervalos sin servicio van con 0. No debe haber valores faltantes.
2. **Topología** (`adyacencia.csv`): matriz N × N de 0/1 sin cabecera, con un 1 entre estaciones consecutivas de
   una misma línea; las estaciones de combinación conectan ambas líneas. Mismo orden que las columnas de flujo.
3. **Configuración**: copiar `t2/datasets/santiago.plantilla.json` como `santiago.json` y completar rutas, fecha de
   inicio y partición. La plantilla trae tres tipos de día (laboral, sábado, domingo/festivo) y feriados chilenos
   automáticos (`"auto:CL"`, requiere `pip install holidays`); ambos son configurables.
4. **Correr**: `bash t2/run_all.sh santiago`. Los resultados quedan en `t2/results/santiago/`.

Recomendaciones para el período: al menos 8 a 10 semanas, para tener varios lunes, sábados y feriados en cada
partición, y un test de dos semanas completas o más. Los datos deben ser agregados por estación e intervalo;
no se necesita ni conviene usar información de tarjetas individuales.

## Cambios de protocolo respecto a la Tarea 1

- Se descarta la fila de cabecera de `HZ_flow.csv` (en el JSON: `"cabecera": true`).
- Partición cronológica por días: en Hangzhou, 17 de entrenamiento, 3 de validación, 5 de test (lunes 21 a viernes 25).
- Escala de normalización calculada sólo con entrenamiento.
- Checkpoint elegido por MAE en validación, tres semillas por variante.
- Todas las variantes se evalúan en las mismas ventanas de test (342 en Hangzhou).
