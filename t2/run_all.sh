#!/usr/bin/env bash
# Corre todos los experimentos de la Tarea 2 para un dataset.
# Uso (desde la raíz del repo):  bash t2/run_all.sh [hangzhou|santiago]
# El dataset se describe en t2/datasets/<nombre>.json. Dos colas en paralelo para aprovechar la GPU;
# en Hangzhou toma ~1 h 40 min en un Apple M5 Pro (MPS).
set -e
DS=${1:-hangzhou}
cd "$(dirname "$0")"
mkdir -p logs/$DS
python linear.py --dataset $DS > logs/$DS/linear.log 2>&1
( for s in 0 1 2; do python train_gcn.py --dataset $DS --variant base --seed $s --epochs 200 > logs/$DS/gcn_base_s$s.log 2>&1; done ) &
( for s in 0 1 2; do python train_gcn.py --dataset $DS --variant ref  --seed $s --epochs 200 > logs/$DS/gcn_ref_s$s.log  2>&1; done ) &
wait
python analyze.py --dataset $DS
