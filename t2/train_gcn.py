# -*- coding: utf-8 -*-
"""Entrena PMR-GCN (código oficial, con la corrección de la KL de la Tarea 1) en dos variantes:

  base : entrada = flujo de las últimas 12 ventanas; salida = flujo de los próximos 4 pasos.
  ref  : entrada = flujo + referencia estacional por tipo de día (2 canales);
         salida = referencia estacional de los objetivos + corrección aprendida por la red (residual).

Mismos hiperparámetros que el paper y el repositorio. El checkpoint se elige por MAE en validación.
Uso:  python t2/train_gcn.py --variant ref --seed 0 --epochs 200 [--dataset hangzhou]
"""
import argparse, json, os, random, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ROOT, HERE, Dataset, add_dataset_arg, metrics

sys.path.insert(0, os.path.join(ROOT, 'PMC-GCN', 'Code'))
sys.path.insert(0, os.path.join(ROOT, 'PMC-GCN', 'Code', 'lib'))
from Model.ST_Transformer_new_sinembedding_with_Two_channel_model import STTransformer_sinembedding  # noqa: E402


def kl_sym(p, q):
    """KL simétrica entre las salidas de ambos canales, en forma estable (log_target=True)."""
    lp, lq = F.log_softmax(p, dim=-1), F.log_softmax(q, dim=-1)
    return 0.5 * (F.kl_div(lp, lq, reduction='none', log_target=True).mean()
                  + F.kl_div(lq, lp, reduction='none', log_target=True).mean())


def tensors(w, scale, variant):
    x = w['x'][:, :, None, :] / scale                                   # (S, N, 1, 12)
    if variant == 'ref':
        x = np.concatenate([x, w['xs'][:, :, None, :] / scale], axis=2)  # (S, N, 2, 12)
    base = w['s'] / scale if variant == 'ref' else np.zeros_like(w['y'])
    return (torch.tensor(x, dtype=torch.float32), torch.tensor(base, dtype=torch.float32),
            torch.tensor(w['y'], dtype=torch.float32))


def predict(net, X, B, scale, device, bs=64):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            o = net(X[i:i + bs].to(device).permute(0, 2, 1, 3)).cpu()
            out.append((o + B[i:i + bs]) * scale)
    return torch.cat(out).numpy()


def main():
    ap = add_dataset_arg(argparse.ArgumentParser())
    ap.add_argument('--variant', choices=['base', 'ref'], required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.004)
    ap.add_argument('--weight_decay', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--alpha', type=float, default=5.0)
    ap.add_argument('--device', default='auto')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = args.device if args.device != 'auto' else (
        'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    ds = Dataset(args.dataset)
    out_dir = os.path.join(HERE, 'runs', ds.name, f'gcn_{args.variant}_s{args.seed}')
    os.makedirs(out_dir, exist_ok=True)

    W = ds.windows()
    scale = ds.train_scale()
    thr = ds.peak_threshold(W)
    Xtr, Btr, Ytr = tensors(W['train'], scale, args.variant)
    Xva, Bva, _ = tensors(W['val'], scale, args.variant)
    Xte, Bte, _ = tensors(W['test'], scale, args.variant)
    g = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Btr, Ytr),
                                         batch_size=args.batch_size, shuffle=True, generator=g)

    A = torch.tensor(ds.adj)
    in_ch = 2 if args.variant == 'ref' else 1
    mk = lambda: STTransformer_sinembedding(A, in_ch, 64, ds.day, 3, ds.seq, ds.pre, 4, 2, device, 4,
                                            dropout=args.dropout).to(device)
    net1, net2 = mk(), mk()
    opt = torch.optim.Adam([{'params': net1.parameters()}, {'params': net2.parameters()}],
                           lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()
    print(f'{ds.title} | device {device} | variante {args.variant} | semilla {args.seed} | '
          f'ventanas {len(Xtr)}/{len(Xva)}/{len(Xte)} | parámetros por red '
          f'{sum(p.numel() for p in net1.parameters())}', flush=True)

    history, best = [], (np.inf, -1)
    t0 = time.time()
    for ep in range(args.epochs):
        net1.train(); net2.train()
        tot = 0.0
        for x, b, y in loader:
            x, b, y = x.to(device), b.to(device), y.to(device)
            opt.zero_grad()
            o1 = (net1(x.permute(0, 2, 1, 3)) + b) * scale
            o2 = (net2(x.permute(0, 2, 1, 3)) + b) * scale
            loss = 0.5 * (mse(o1, y) + mse(o2, y)) + args.alpha * kl_sym(o1, o2)
            loss.backward(); opt.step()
            tot += loss.item()
        pva = predict(net1, Xva, Bva, scale, device)
        pte = predict(net1, Xte, Bte, scale, device)
        mva = metrics(W['val']['y'], pva)['all']
        mte = metrics(W['test']['y'], pte, thr)
        rec = dict(epoch=ep, train_loss=tot, val=mva, test=mte['all'], test_peak=mte['peak']['MAE'],
                   time=time.time() - t0)
        history.append(rec)
        if mva['MAE'] < best[0]:
            best = (mva['MAE'], ep)
            torch.save(net1.state_dict(), os.path.join(out_dir, 'best.pt'))
            np.save(os.path.join(out_dir, 'test_pred.npy'), pte)
        print(f"ep {ep:3d} | loss {tot:11.1f} | val MAE {mva['MAE']:6.2f} | test MAE {mte['all']['MAE']:6.2f} "
              f"| {rec['time']:5.0f}s", flush=True)
        with open(os.path.join(out_dir, 'history.json'), 'w') as f:
            json.dump(history, f)

    pte = np.load(os.path.join(out_dir, 'test_pred.npy'))
    summary = dict(args=vars(args), device=device, torch=torch.__version__, best_epoch=best[1],
                   best_val_MAE=best[0], minutes=(time.time() - t0) / 60,
                   test=metrics(W['test']['y'], pte, thr))
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nmejor época por validación: {best[1]} | test MAE {summary['test']['all']['MAE']:.2f} "
          f"| {summary['minutes']:.1f} min", flush=True)


if __name__ == '__main__':
    main()
