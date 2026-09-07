# -*- coding: utf-8 -*-
"""Baselines triviales sobre exactamente el mismo split/ventanas de test que usa el código oficial
(generate_dataset: split 80/20 cronológico, seq_len=12, pre_len=4). Sirven de referencia para
juzgar si el modelo aporta valor sobre predictores ingenuos."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, '..', 'PMC-GCN')
sys.path.insert(0, os.path.join(REPO, 'Code', 'lib'))
from utils import generate_dataset, load_features, masked_mape_np

def metrics(tgt, pred):
    out = {}
    for i in range(tgt.shape[-1]):
        t, p = tgt[..., i], pred[..., i]
        out[f'h{i+1}'] = dict(MAE=float(np.abs(t-p).mean()), RMSE=float(np.sqrt(((t-p)**2).mean())), MAPE=float(masked_mape_np(t, p, 0)))
    out['all'] = dict(MAE=float(np.abs(tgt-pred).mean()), RMSE=float(np.sqrt(((tgt-pred)**2).mean())), MAPE=float(masked_mape_np(tgt.reshape(-1,1), pred.reshape(-1,1), 0)))
    return out

def main(drop_header=False):
    feat = load_features(os.path.join(REPO, 'Dataset', 'HZ', 'HZ_flow.csv'))
    if drop_header: feat = feat[1:]
    seq_len, pre_len, T = 12, 4, 69
    # NB: generate_dataset(normalize=False) del repo lanza UnboundLocalError (max_val no definido); se usa normalize=True y se desnormaliza
    trX, trY, teX, teY, max_val = generate_dataset(feat, seq_len, pre_len, None, 0.8, normalize=True)
    teX, teY = teX * max_val, teY * max_val
    # teX: (S, 12, N), teY: (S, 4, N)  -> pasar a (S, N, 4) como el modelo
    teY_ = teY.transpose(0, 2, 1); res = {}
    # 1) Último valor observado repetido
    res['last_value'] = metrics(teY_, np.repeat(teX[:, -1:, :], pre_len, 1).transpose(0, 2, 1))
    # 2) Media de la ventana de entrada
    res['window_mean'] = metrics(teY_, np.repeat(teX.mean(1, keepdims=True), pre_len, 1).transpose(0, 2, 1))
    # 3) Naive estacional: mismo intervalo del día anterior (lag 69)
    n_train = int(feat.shape[0] * 0.8); test = feat[n_train:]
    full = feat; start = n_train + seq_len          # índice absoluto del primer target de test
    S = teY.shape[0]
    seas = np.stack([full[start + s - T : start + s - T + pre_len] for s in range(S)])
    res['seasonal_naive_1day'] = metrics(teY_, seas.transpose(0, 2, 1))
    # 4) Historical average por intervalo del día calculado sólo con train (asume 69 intervalos/día)
    tr = feat[:n_train]; idx = np.arange(feat.shape[0]) % T
    ha = np.stack([tr[idx[:n_train] == k].mean(0) for k in range(T)])          # (69, N)
    ha_pred = np.stack([ha[idx[start + s : start + s + pre_len]] for s in range(S)])
    res['historical_average'] = metrics(teY_, ha_pred.transpose(0, 2, 1))
    res['_info'] = dict(n_test=int(S), max_val=float(max_val), drop_header=drop_header, target_mean=float(teY.mean()))
    return res

if __name__ == '__main__':
    r = main(drop_header='--drop_header_row' in sys.argv)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    with open(os.path.join(HERE, 'results', 'baselines.json'), 'w') as f: json.dump(r, f, indent=2)
    print(f"test samples {r['_info']['n_test']}, target mean {r['_info']['target_mean']:.1f}")
    for k, v in r.items():
        if k.startswith('_'): continue
        print(f"{k:22s} all: MAE {v['all']['MAE']:7.2f} RMSE {v['all']['RMSE']:7.2f} MAPE {v['all']['MAPE']:6.2f} | h1 MAE {v['h1']['MAE']:6.2f} h4 MAE {v['h4']['MAE']:6.2f}")
