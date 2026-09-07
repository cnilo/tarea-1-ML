# -*- coding: utf-8 -*-
"""Genera tablas y figuras a partir de runs/*/history.json, summary.json y results/baselines.json."""
import os, json, glob, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs'); OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)

def load(run):
    h = json.load(open(os.path.join(run, 'history.json')))
    s = json.load(open(os.path.join(run, 'summary.json'))) if os.path.exists(os.path.join(run, 'summary.json')) else None
    return h, s

runs = {os.path.basename(r): load(r) for r in sorted(glob.glob(os.path.join(RUNS, '*'))) if os.path.exists(os.path.join(r, 'history.json'))}
base = json.load(open(os.path.join(OUT, 'baselines.json')))
rows = []
for name, (h, s) in runs.items():
    ev = [r for r in h if 'test_net1' in r]
    mae = np.array([r['test_net1']['all']['MAE'] for r in ev]); ep = np.array([r['epoch'] for r in ev])
    last = ev[-1]['test_net1']
    best_test_i = int(mae.argmin())
    rec = dict(run=name, epochs=len(h), time_min=h[-1]['time']/60,
               best_train_epoch=s['best_epoch'] if s else int(np.argmin([r['train_loss'] for r in h])),
               final=s['final_test_best_epoch'] if s else None,
               last=last, oracle_epoch=int(ep[best_test_i]), oracle=ev[best_test_i]['test_net1'],
               mean_last50=dict(MAE=float(mae[-50:].mean()), std_last50=float(mae[-50:].std())),
               at_ep199=(ev[[r['epoch'] for r in ev].index(199)]['test_net1'] if 199 in [r['epoch'] for r in ev] else None))
    rows.append(rec)
    # fig: curvas
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].semilogy([r['epoch'] for r in h], [r['train_loss'] for r in h]); ax[0].set_title('Pérdida de entrenamiento (suma por época)'); ax[0].set_xlabel('época')
    ax[1].plot(ep, mae, lw=1, label='MAE test (net1)')
    if 'test_net2' in ev[0]: ax[1].plot(ep, [r['test_net2']['all']['MAE'] for r in ev], lw=1, alpha=.7, label='MAE test (net2)')
    for k, c in [('seasonal_naive_1day', 'gray'), ('historical_average', 'k')]:
        ax[1].axhline(base[k]['all']['MAE'], ls='--', c=c, lw=1, label=k)
    ax[1].set_ylim(0, max(150, mae[-1]*1.5)); ax[1].set_title('MAE en test por época'); ax[1].set_xlabel('época'); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'curves_{name}.png'), dpi=160); plt.close(fig)

json.dump(rows, open(os.path.join(OUT, 'runs_summary.json'), 'w'), indent=2)
print(f"{'run':30s} {'ep':>4s} {'min':>6s} | best_train_ep  MAE   RMSE   MAPE | oracle_ep  MAE | last MAE | mean±std last50")
for r in rows:
    f = r['final'] or r['last']
    print(f"{r['run']:30s} {r['epochs']:4d} {r['time_min']:6.0f} | {r['best_train_epoch']:5d} {f['all']['MAE']:8.2f} {f['all']['RMSE']:6.2f} {f['all']['MAPE']:6.2f} | {r['oracle_epoch']:5d} {r['oracle']['all']['MAE']:6.2f} | {r['last']['all']['MAE']:8.2f} | {r['mean_last50']['MAE']:.2f}±{r['mean_last50']['std_last50']:.2f}")
print('\nbaselines:'); [print(f"  {k:22s} MAE {v['all']['MAE']:7.2f} RMSE {v['all']['RMSE']:7.2f} MAPE {v['all']['MAPE']:6.2f}") for k, v in base.items() if not k.startswith('_')]

# fig: predicción vs real para el run principal, si existe
main = [r for r in sorted(glob.glob(os.path.join(RUNS, 'HZ_seed0'))) if os.path.exists(os.path.join(r, 'test_predictions.npz'))]
if main:
    d = np.load(os.path.join(main[0], 'test_predictions.npz')); pred, tgt = d['prediction'], d['target']  # (S, N, 4)
    st = int(np.argmax(tgt[:, :, 0].mean(0)))  # estación de mayor flujo
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(tgt[:, st, 0], label='real', lw=1); ax.plot(pred[:, st, 0], label='PMR-GCN (h=1, 15 min)', lw=1)
    ax.set_title(f'Estación {st+1} (mayor flujo): horizonte 15 min, conjunto de test'); ax.set_xlabel('paso de 15 min'); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'pred_vs_real.png'), dpi=160)
    # métricas por horizonte
    hz = json.load(open(os.path.join(main[0], 'summary.json')))['final_test_best_epoch']
    print('\npor horizonte (run principal):', {k: {m: round(v, 2) for m, v in hz[k].items()} for k in hz})
    # error en hora punta vs resto
    err = np.abs(pred - tgt); hi = tgt > np.percentile(tgt, 90)
    print(f"MAE en el 10% de valores más altos (peaks): {err[hi].mean():.2f} | resto: {err[~hi].mean():.2f}")
    json.dump(dict(mae_top10=float(err[hi].mean()), mae_rest=float(err[~hi].mean()), p90=float(np.percentile(tgt, 90)), station=st+1), open(os.path.join(OUT, "peaks.json"), "w"), indent=2)
