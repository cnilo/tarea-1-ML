# -*- coding: utf-8 -*-
"""Modelo de control: regresión ridge por estación, con y sin la referencia estacional por tipo de día.

Sirve para separar dos efectos: cuánto aporta la *información* (la referencia) y cuánto la *arquitectura*
(PMR-GCN frente a un modelo lineal sin grafo). Se ajusta un modelo por estación con salida múltiple
(4 horizontes); la regularización se elige por MAE en validación. También calcula los predictores ingenuos.
Uso:  python t2/linear.py [--dataset hangzhou]
"""
import argparse, json, os
import numpy as np
from sklearn.linear_model import Ridge

from common import HERE, Dataset, add_dataset_arg, metrics

ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]


def features(w, n, use_ref):
    f = [w['x'][:, n, :]]
    if use_ref:
        f += [w['xs'][:, n, :], w['s'][:, n, :]]
    return np.concatenate(f, axis=1)


def fit_predict(W, use_ref):
    N = W['train']['x'].shape[1]
    best = None
    for a in ALPHAS:
        pv = np.zeros_like(W['val']['y'])
        for n in range(N):
            m = Ridge(alpha=a).fit(features(W['train'], n, use_ref), W['train']['y'][:, n, :])
            pv[:, n, :] = m.predict(features(W['val'], n, use_ref))
        mae = np.abs(pv - W['val']['y']).mean()
        if best is None or mae < best[1]:
            best = (a, mae)
    a = best[0]
    pt = np.zeros_like(W['test']['y'])
    for n in range(N):
        m = Ridge(alpha=a).fit(features(W['train'], n, use_ref), W['train']['y'][:, n, :])
        pt[:, n, :] = m.predict(features(W['test'], n, use_ref))
    return pt, a, best[1]


def main():
    args = add_dataset_arg(argparse.ArgumentParser()).parse_args()
    ds = Dataset(args.dataset)
    W = ds.windows()
    thr = ds.peak_threshold(W)
    te = W['test']
    out_dir = os.path.join(HERE, 'runs', ds.name)
    os.makedirs(out_dir, exist_ok=True)
    preds = {
        'naive_last': te['last'],
        'naive_ayer': te['ayer'],
        'naive_tipo': te['s'],
        'naive_ha': ds.historical_average(te['t']),
        'naive_ha_tipo': ds.historical_average(te['t'], by_type=True),
    }
    info = {}
    for use_ref in (False, True):
        key = 'ridge_ref' if use_ref else 'ridge_base'
        preds[key], a, vmae = fit_predict(W, use_ref)
        info[key] = dict(alpha=a, val_MAE=float(vmae))
    res = {k: metrics(te['y'], p, thr) for k, p in preds.items()}
    np.savez(os.path.join(out_dir, 'linear_naive_preds.npz'), y=te['y'], t=te['t'], **preds)
    with open(os.path.join(out_dir, 'linear_naive.json'), 'w') as f:
        json.dump(dict(results=res, ridge=info, peak_threshold=thr), f, indent=2)
    for k, m in res.items():
        print(f"{k:15s} MAE {m['all']['MAE']:7.2f}  RMSE {m['all']['RMSE']:7.2f}  "
              f"MAPE {m['all']['MAPE']:6.2f}  peak MAE {m['peak']['MAE']:6.1f}", flush=True)
    print('ridge:', info)


if __name__ == '__main__':
    main()
