# -*- coding: utf-8 -*-
"""Consolida los resultados de la Tarea 2: tabla principal, intervalos de confianza y figuras.
Uso:  python t2/analyze.py [--dataset hangzhou]    (después de linear.py y train_gcn.py)
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import HERE, Dataset, add_dataset_arg, block_bootstrap_diff, metrics

ARGS = add_dataset_arg(argparse.ArgumentParser()).parse_args()
DS = Dataset(ARGS.dataset)
RUNS, OUT = os.path.join(HERE, 'runs', DS.name), os.path.join(HERE, 'results', DS.name)
os.makedirs(OUT, exist_ok=True)
C = dict(ink='#22252A', grey='#8A8F96', red='#C8102E', yellow='#EFAB00', green='#00803C', blue='#0B4EA2')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': C['ink'], 'axes.labelcolor': C['ink'], 'xtick.color': C['ink'],
                     'ytick.color': C['ink'], 'figure.dpi': 160})

LN = json.load(open(os.path.join(RUNS, 'linear_naive.json')))
P = dict(np.load(os.path.join(RUNS, 'linear_naive_preds.npz')))
y, ts, thr = P.pop('y'), P.pop('t'), LN['peak_threshold']


def load_gcn(variant):
    runs = sorted(glob.glob(os.path.join(RUNS, f'gcn_{variant}_s*')))
    runs = [r for r in runs if os.path.exists(os.path.join(r, 'summary.json'))]
    preds = [np.load(os.path.join(r, 'test_pred.npy')) for r in runs]
    summ = [json.load(open(os.path.join(r, 'summary.json'))) for r in runs]
    hist = [json.load(open(os.path.join(r, 'history.json'))) for r in runs]
    return preds, summ, hist


G = {v: load_gcn(v) for v in ('base', 'ref')}
LABEL = {
    'naive_last': 'Ingenuo: último valor', 'naive_ayer': 'Ingenuo: mismo intervalo de ayer',
    'naive_ha': 'Ingenuo: promedio histórico', 'naive_ha_tipo': 'Ingenuo: promedio histórico por tipo de día',
    'naive_tipo': 'Ingenuo: último día del mismo tipo',
    'ridge_base': 'Ridge sin referencia', 'ridge_ref': 'Ridge con referencia',
    'gcn_base': 'PMR-GCN sin referencia', 'gcn_ref': 'PMR-GCN con referencia',
}

# ---------- tabla principal ----------
rows = {}
for k in ['naive_last', 'naive_ayer', 'naive_ha', 'naive_ha_tipo', 'naive_tipo', 'ridge_base', 'ridge_ref']:
    m = LN['results'][k]
    rows[k] = dict(MAE=m['all']['MAE'], RMSE=m['all']['RMSE'], MAPE=m['all']['MAPE'],
                   peak=m['peak']['MAE'], offpeak=m['offpeak']['MAE'], sd=None, n=1,
                   h=[m[f'h{i}']['MAE'] for i in range(1, 5)])
for v in ('base', 'ref'):
    preds, summ, _ = G[v]
    if not preds:
        continue
    ms = [metrics(y, p, thr) for p in preds]
    rows[f'gcn_{v}'] = dict(
        MAE=float(np.mean([m['all']['MAE'] for m in ms])), RMSE=float(np.mean([m['all']['RMSE'] for m in ms])),
        MAPE=float(np.mean([m['all']['MAPE'] for m in ms])), peak=float(np.mean([m['peak']['MAE'] for m in ms])),
        offpeak=float(np.mean([m['offpeak']['MAE'] for m in ms])),
        sd=float(np.std([m['all']['MAE'] for m in ms], ddof=1)) if len(ms) > 1 else None, n=len(ms),
        per_seed=[m['all']['MAE'] for m in ms], best_epochs=[s['best_epoch'] for s in summ],
        minutes=[s['minutes'] for s in summ],
        h=[float(np.mean([m[f'h{i}']['MAE'] for m in ms])) for i in range(1, 5)])


def abs_err(key):
    """Error absoluto por ventana; para PMR-GCN, promedio sobre semillas (rendimiento esperado de una corrida)."""
    if key.startswith('gcn_'):
        return np.mean([np.abs(p - y) for p in G[key[4:]][0]], axis=0)
    return np.abs(P[key] - y)


# ---------- comparaciones de la hipótesis (IC 95 % por bootstrap de bloques) ----------
comparisons = [('gcn_ref', 'gcn_base'), ('gcn_ref', 'naive_tipo'), ('ridge_ref', 'naive_tipo'),
               ('gcn_ref', 'ridge_ref'), ('ridge_ref', 'ridge_base')]
ci = {}
for a, b in comparisons:
    if a in rows and b in rows:
        d, lo, hi = block_bootstrap_diff(abs_err(a), abs_err(b))
        ea, eb = abs_err(a), abs_err(b)
        pk = y > thr
        ci[f'{a}__vs__{b}'] = dict(diff=d, lo=lo, hi=hi, rel=d / rows[b]['MAE'] * 100,
                                   peak_diff=float(ea[pk].mean() - eb[pk].mean()))

# ---------- PMR-GCN con referencia frente al ingenuo, semilla por semilla ----------
# El IC anterior usa el error promediado entre semillas y captura la variabilidad del test, no la del entrenamiento.
per_seed_vs_naive = []
if G['ref'][0]:
    en = abs_err('naive_tipo')
    for p in G['ref'][0]:
        d, lo, hi = block_bootstrap_diff(np.abs(p - y), en)
        per_seed_vs_naive.append(dict(diff=d, lo=lo, hi=hi))

# ---------- MAE por día de test ----------
day = (ts + 1) // DS.day
per_day = {k: [float(abs_err(k)[day == d].mean()) for d in np.unique(day)] for k in rows}
days = [DS.day_label(d) for d in np.unique(day)]

json.dump(dict(rows=rows, ci=ci, per_seed_vs_naive=per_seed_vs_naive, per_day=per_day, days=days, peak_threshold=thr,
               n_test_windows=int(len(ts))), open(os.path.join(OUT, 'summary.json'), 'w'), indent=2)

print(f"{'método':46s} {'MAE':>12s} {'RMSE':>7s} {'MAPE':>6s} {'peak':>7s} {'resto':>6s}")
for k, r in sorted(rows.items(), key=lambda kv: -kv[1]['MAE']):
    sd = f"±{r['sd']:.2f}" if r['sd'] is not None else ''
    print(f"{LABEL[k]:46s} {r['MAE']:6.2f}{sd:>6s} {r['RMSE']:7.2f} {r['MAPE']:6.2f} {r['peak']:7.1f} {r['offpeak']:6.1f}")
print()
for k, c in ci.items():
    a, b = k.split('__vs__')
    print(f"{LABEL[a]} − {LABEL[b]}: {c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] ({c['rel']:+.1f} %), peaks {c['peak_diff']:+.1f}")
for i, c in enumerate(per_seed_vs_naive):
    print(f"  semilla {i}: PMR-GCN con referencia − ingenuo por tipo: {c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]")
print('\nMAE por día:', days)
for k in rows:
    print(f"  {LABEL[k]:46s}", ' '.join(f'{v:6.1f}' for v in per_day[k]))
for v in ('base', 'ref'):
    if f'gcn_{v}' in rows:
        r = rows[f'gcn_{v}']
        print(f"gcn_{v}: semillas {np.round(r['per_seed'], 2)} épocas elegidas {r['best_epochs']} minutos {np.round(r['minutes'], 1)}")

# ---------- figuras ----------
order = ['naive_ayer', 'naive_ha', 'naive_ha_tipo', 'naive_tipo', 'gcn_base', 'ridge_base', 'ridge_ref', 'gcn_ref']
order = [k for k in order if k in rows]
col = {k: (C['grey'] if k.startswith('naive') else C['yellow'] if k.startswith('ridge') else C['blue']) for k in order}
fig, ax = plt.subplots(figsize=(7.2, 3.6))
vals = [rows[k]['MAE'] for k in order]
err = [rows[k]['sd'] or 0 for k in order]
ax.barh(range(len(order)), vals, xerr=err, color=[col[k] for k in order], height=.62, capsize=3)
if DS.reference:
    ax.axvline(DS.reference['mae'], color=C['red'], ls='--', lw=1.2)
    ax.text(DS.reference['mae'], len(order) - .45, f" {DS.reference['etiqueta']} ({DS.reference['mae']:.1f})",
            color=C['red'], fontsize=8, va='center')
ax.set_yticks(range(len(order)), [LABEL[k] for k in order], fontsize=8.5)
for i, v in enumerate(vals):
    ax.text(v + 1.5 + err[i], i, f'{v:.1f}', va='center', fontsize=8.5)
ax.set_xlabel('MAE en test (pasajeros por intervalo de 15 min)')
ax.invert_yaxis()
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_mae.png')); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.2))
show = [k for k in ['naive_ayer', 'naive_tipo', 'gcn_base', 'ridge_ref', 'gcn_ref'] if k in rows]
sty = {'naive_ayer': (C['grey'], ':'), 'naive_tipo': (C['ink'], '--'), 'gcn_base': (C['blue'], ':'),
       'ridge_ref': (C['yellow'], '-'), 'gcn_ref': (C['blue'], '-')}
for k in show:
    ax.plot(range(len(days)), per_day[k], marker='o', color=sty[k][0], ls=sty[k][1], label=LABEL[k], lw=1.6)
ax.set_xticks(range(len(days)), days, fontsize=8 if len(days) <= 7 else 6, rotation=0 if len(days) <= 7 else 90)
ax.set_ylabel('MAE en test'); ax.legend(fontsize=7.5, frameon=False, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_por_dia.png')); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=False)
for ax, part in zip(axs, ['peak', 'offpeak']):
    ks = [k for k in ['naive_tipo', 'gcn_base', 'ridge_ref', 'gcn_ref'] if k in rows]
    ax.bar(range(len(ks)), [rows[k][part] for k in ks], color=[col.get(k, C['ink']) if k != 'naive_tipo' else C['grey'] for k in ks])
    SHORT = {'naive_tipo': 'Ingenuo\ntipo de día', 'gcn_base': 'PMR-GCN\nsin ref.', 'ridge_ref': 'Ridge\ncon ref.', 'gcn_ref': 'PMR-GCN\ncon ref.'}
    ax.set_xticks(range(len(ks)), [SHORT[k] for k in ks], fontsize=7.5)
    ax.set_title('Peaks (10 % de mayor flujo)' if part == 'peak' else 'Resto de los intervalos', fontsize=9)
    for i, k in enumerate(ks):
        ax.text(i, rows[k][part], f'{rows[k][part]:.0f}', ha='center', va='bottom', fontsize=8)
axs[0].set_ylabel('MAE en test')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_peaks.png')); plt.close(fig)

if G['base'][2] and G['ref'][2]:
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for v, c in (('base', C['grey']), ('ref', C['blue'])):
        H = G[v][2]
        n = min(len(h) for h in H)
        val = np.array([[r['val']['MAE'] for r in h[:n]] for h in H])
        ax.plot(val.mean(0), color=c, lw=1.4, label=f'PMR-GCN {"con" if v == "ref" else "sin"} referencia')
        ax.fill_between(range(n), val.min(0), val.max(0), color=c, alpha=.18)
    Wv = DS.windows()['val']
    naive_val = float(np.abs(Wv['s'] - Wv['y']).mean())
    ax.axhline(naive_val, color=C['ink'], ls='--', lw=1)
    ax.text(n - 1, naive_val + 3, f'ingenuo por tipo de día en validación ({naive_val:.1f})', ha='right', fontsize=8)
    ax.set_ylim(0, 160); ax.set_xlabel('época'); ax.set_ylabel('MAE en validación')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_curvas.png')); plt.close(fig)
print('\nfiguras en', OUT)

# ---------- números para el informe (macros LaTeX) ----------
def word(k):
    return {'naive_last': 'NLast', 'naive_ayer': 'NAyer', 'naive_ha': 'NHA', 'naive_ha_tipo': 'NHATipo',
            'naive_tipo': 'NTipo', 'ridge_base': 'RidgeBase', 'ridge_ref': 'RidgeRef',
            'gcn_base': 'GcnBase', 'gcn_ref': 'GcnRef'}[k]


def f2(x):
    return f'{x:.1f}'


lines = ['% generado por t2/analyze.py; no editar a mano']
for k, r in rows.items():
    w = word(k)
    for m in ('MAE', 'RMSE', 'MAPE', 'peak', 'offpeak'):
        lines.append(f'\\newcommand{{\\{w}{m.capitalize()}}}{{{f2(r[m])}}}')
    for i, v in enumerate(r['h']):
        lines.append(f'\\newcommand{{\\{w}H{"abcd"[i]}}}{{{f2(v)}}}')
    if r['sd'] is not None:
        lines.append(f'\\newcommand{{\\{w}Sd}}{{{r["sd"]:.1f}}}')
        lines.append(f'\\newcommand{{\\{w}Seeds}}{{{", ".join(f2(v) for v in r["per_seed"])}}}')
        lines.append(f'\\newcommand{{\\{w}Epochs}}{{{", ".join(str(e) for e in r["best_epochs"])}}}')
        lines.append(f'\\newcommand{{\\{w}Minutes}}{{{np.mean(r["minutes"]):.0f}}}')
for k, c in ci.items():
    a, b = k.split('__vs__')
    w = word(a) + 'Vs' + word(b)
    lines.append(f'\\newcommand{{\\{w}Diff}}{{{c["diff"]:+.1f}}}')
    lines.append(f'\\newcommand{{\\{w}Lo}}{{{c["lo"]:+.1f}}}')
    lines.append(f'\\newcommand{{\\{w}Hi}}{{{c["hi"]:+.1f}}}')
    lines.append(f'\\newcommand{{\\{w}Rel}}{{{c["rel"]:+.1f}}}')
import unicodedata
def macro_day(lbl, seen):
    base = unicodedata.normalize('NFKD', lbl.split()[0]).encode('ascii', 'ignore').decode()
    seen[base] = seen.get(base, 0) + 1
    return base if seen[base] == 1 else base + 'abcdefghijklmnopqrstuvwxyz'[seen[base] - 2]
for k in rows:
    seen = {}
    for d, v in zip(days, per_day[k]):
        lines.append(f'\\newcommand{{\\{word(k)}Dia{macro_day(d, seen)}}}{{{f2(v)}}}')
for i, c in enumerate(per_seed_vs_naive):
    tag = 'abc'[i]
    lines.append(f'\\newcommand{{\\SeedVsNTipo{tag}}}{{{c["diff"]:+.1f} [{c["lo"]:+.1f}, {c["hi"]:+.1f}]}}')
lines.append(f'\\newcommand{{\\SeedsBeatNaive}}{{{sum(c["hi"] < 0 for c in per_seed_vs_naive)}}}')
lines.append(f'\\newcommand{{\\NSeeds}}{{{len(per_seed_vs_naive)}}}')
lines.append(f'\\newcommand{{\\PeakThr}}{{{thr:.0f}}}')
lines.append(f'\\newcommand{{\\NTest}}{{{len(ts)}}}')
lines.append(f'\\newcommand{{\\DatasetNombre}}{{{DS.title}}}')
open(os.path.join(OUT, 'numeros.tex'), 'w').write('\n'.join(lines) + '\n')
print('macros en', os.path.join(OUT, 'numeros.tex'))
