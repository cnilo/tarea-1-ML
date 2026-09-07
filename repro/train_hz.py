# -*- coding: utf-8 -*-
"""
Reproducción de PMR-GCN (Gao et al., IEEE T-ITS 2024) sobre el dataset Hangzhou (HZ)
incluido en el repositorio oficial https://github.com/cgao-comp/PMC-GCN.

Este script replica fielmente `Code/New_train_batch_with_two_channel.py` y
`Code/predict_batch_with_two_channel.py` del repo (mismo modelo, misma pérdida,
mismos hiperparámetros), pero:
  * usa la configuración de HZ que en el repo está comentada (time_num=69, rutas HZ),
  * fija semillas, acepta argumentos por CLI y soporta CPU / CUDA / MPS (Apple Silicon),
  * registra métricas de test en cada época (para analizar el criterio de selección
    del modelo, que en el repo es la pérdida de entrenamiento), y
  * no depende de tensorboardX.
"""
import argparse, json, os, sys, time, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, '..', 'PMC-GCN')
sys.path.insert(0, os.path.join(REPO, 'Code'))
sys.path.insert(0, os.path.join(REPO, 'Code', 'lib'))
from Model.ST_Transformer_new_sinembedding_with_Two_channel_model import STTransformer_sinembedding  # noqa
from utils import generate_torch_datasets, load_features, masked_mape_np  # noqa


def compute_kl_loss(p, q, impl='log_target'):
    """KL simétrica entre las salidas de las dos redes (estilo R-Drop).

    impl='original'  : copia literal de New_train_batch_with_two_channel.py. Con los datos HZ
                       (salidas en escala de pasajeros, hasta ~1e4) el softmax se hace exactamente 0
                       en ~74% de las entradas y el gradiente respecto al target es NaN en el
                       primer paso de entrenamiento (verificado con torch 2.14).
    impl='log_target': misma cantidad matemática, calculada con log_target=True (estable).
    """
    if impl == 'original':
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none').mean()
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none').mean()
    else:
        lp, lq = F.log_softmax(p, dim=-1), F.log_softmax(q, dim=-1)
        p_loss = F.kl_div(lp, lq, reduction='none', log_target=True).mean()
        q_loss = F.kl_div(lq, lp, reduction='none', log_target=True).mean()
    return (p_loss + q_loss) / 2


def evaluate(net, loader, max_val, device):
    """Réplica de predict_and_save_results_my: MAE/RMSE/MAPE por horizonte y global."""
    net.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = net(x.permute(0, 2, 1, 3))
            preds.append(out.cpu().numpy()); targets.append(y.numpy())
    pred = np.concatenate(preds, 0) * max_val
    tgt = np.concatenate(targets, 0) * max_val
    res = {}
    for i in range(pred.shape[2]):
        res[f'h{i+1}'] = dict(
            MAE=float(np.mean(np.abs(tgt[:, :, i] - pred[:, :, i]))),
            RMSE=float(np.sqrt(np.mean((tgt[:, :, i] - pred[:, :, i]) ** 2))),
            MAPE=float(masked_mape_np(tgt[:, :, i], pred[:, :, i], 0)))
    res['all'] = dict(
        MAE=float(np.mean(np.abs(tgt - pred))),
        RMSE=float(np.sqrt(np.mean((tgt - pred) ** 2))),
        MAPE=float(masked_mape_np(tgt.reshape(-1, 1), pred.reshape(-1, 1), 0)))
    return res, pred, tgt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='HZ')
    ap.add_argument('--epochs', type=int, default=800)          # repo: 800
    ap.add_argument('--batch_size', type=int, default=16)       # repo: 16
    ap.add_argument('--lr', type=float, default=0.004)          # repo: 0.004 (SH, HZ, CQ)
    ap.add_argument('--weight_decay', type=float, default=0.001)
    ap.add_argument('--alpha', type=float, default=5.0)         # peso de la KL (repo: 5)
    ap.add_argument('--dropout', type=float, default=0.2)       # repo: 0.2 en entrenamiento
    ap.add_argument('--time_num', type=int, default=69)         # HZ: 69 intervalos/día (comentado en el repo)
    ap.add_argument('--seq_len', type=int, default=12)
    ap.add_argument('--pre_len', type=int, default=4)
    ap.add_argument('--split_ratio', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--drop_header_row', action='store_true',
                    help='descarta la 1ª fila del CSV (1..81), que el código original lee como datos')
    ap.add_argument('--single_channel', action='store_true',
                    help='ablación: sin la 2ª red ni la regularización KL')
    ap.add_argument('--loss', default='mse', choices=['mse', 'mae'], help='mse = código oficial; mae = Ec. (10)/(16) del paper')
    ap.add_argument('--kl_impl', default='log_target', choices=['original', 'log_target'])
    ap.add_argument('--eval_every', type=int, default=1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = args.device
    print('device:', device, '| torch', torch.__version__)

    out_dir = args.out or os.path.join(HERE, 'runs', f'{args.dataset}_seed{args.seed}' + ('_1ch' if args.single_channel else '') + ('_nohdr' if args.drop_header_row else '') + ('_mae' if args.loss == 'mae' else ''))
    os.makedirs(out_dir, exist_ok=True)

    # ---------- datos (idéntico a repo: normaliza por el máximo global, split 80/20 cronológico) ----------
    ddir = os.path.join(REPO, 'Dataset', args.dataset)
    feat = load_features(os.path.join(ddir, f'{args.dataset}_flow.csv'))
    if args.drop_header_row:
        feat = feat[1:]
    print('flow matrix:', feat.shape, 'max:', feat.max())
    test_ds, train_ds, _, _, _, test_target, max_val = generate_torch_datasets(
        feat, args.seq_len, args.pre_len, None, args.split_ratio, True)
    g = torch.Generator(); g.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print('train samples:', len(train_ds), 'test samples:', len(test_ds))

    A = torch.Tensor(np.array(pd.read_csv(os.path.join(ddir, f'{args.dataset}_adj.csv'), header=None)))

    # ---------- modelo (hiperparámetros del repo) ----------
    mk = lambda dp: STTransformer_sinembedding(A, 1, 64, args.time_num, 3, args.seq_len, args.pre_len, 4, 2, device, 4, dropout=dp).to(device)
    net1 = mk(args.dropout)
    net2 = None if args.single_channel else mk(args.dropout)
    n_params = sum(p.numel() for p in net1.parameters())
    print('params per network:', n_params)

    groups = [{"params": net1.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}]
    if net2 is not None:
        groups.append({"params": net2.parameters(), "lr": args.lr, "weight_decay": args.weight_decay})
    optimizer = torch.optim.Adam(groups)
    criterion = nn.MSELoss() if args.loss == 'mse' else nn.L1Loss()

    history, best_train_loss, best_epoch = [], np.inf, -1
    t0 = time.time()
    for epoch in range(args.epochs):
        net1.train()
        if net2 is not None: net2.train()
        tr_loss_all, kl_all = 0.0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device) * max_val
            optimizer.zero_grad()
            o1 = net1(x.permute(0, 2, 1, 3)) * max_val
            if net2 is None:
                loss = criterion(o1, y); kl = torch.zeros(())
            else:
                o2 = net2(x.permute(0, 2, 1, 3)) * max_val
                ce = 0.5 * (criterion(o1, y) + criterion(o2, y))
                kl = compute_kl_loss(o1, o2, args.kl_impl)
                loss = ce + args.alpha * kl
            loss.backward(); optimizer.step()
            tr_loss_all += loss.item(); kl_all += float(kl.detach())
        rec = dict(epoch=epoch, train_loss=tr_loss_all, kl=kl_all, time=time.time() - t0)
        # selección de modelo tal como el repo: menor pérdida de entrenamiento acumulada
        if tr_loss_all < best_train_loss:
            best_train_loss, best_epoch = tr_loss_all, epoch
            torch.save(net1.state_dict(), os.path.join(out_dir, 'best_net1.pt'))
            if net2 is not None: torch.save(net2.state_dict(), os.path.join(out_dir, 'best_net2.pt'))
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            m1, _, _ = evaluate(net1, test_loader, max_val, device)
            rec['test_net1'] = m1
            if net2 is not None:
                m2, _, _ = evaluate(net2, test_loader, max_val, device); rec['test_net2'] = m2
            print(f"ep {epoch:3d} loss {tr_loss_all:10.1f} kl {kl_all:8.4f} | test net1 MAE {m1['all']['MAE']:.2f} RMSE {m1['all']['RMSE']:.2f} MAPE {m1['all']['MAPE']:.2f} | {rec['time']:.0f}s", flush=True)
        history.append(rec)
        with open(os.path.join(out_dir, 'history.json'), 'w') as f: json.dump(history, f)

    # ---------- evaluación final con el checkpoint de "best epoch" (como predict_batch_with_two_channel.py) ----------
    net1.load_state_dict(torch.load(os.path.join(out_dir, 'best_net1.pt')))
    final, pred, tgt = evaluate(net1, test_loader, max_val, device)
    np.savez(os.path.join(out_dir, 'test_predictions.npz'), prediction=pred, target=tgt)
    summary = dict(args=vars(args), device=device, torch=torch.__version__, params_per_net=n_params,
                   n_train=len(train_ds), n_test=len(test_ds), max_val=float(max_val),
                   best_epoch=best_epoch, best_train_loss=best_train_loss,
                   total_time_s=time.time() - t0, final_test_best_epoch=final)
    with open(os.path.join(out_dir, 'summary.json'), 'w') as f: json.dump(summary, f, indent=2)
    print('\nbest epoch:', best_epoch, '| total time: %.1f min' % ((time.time() - t0) / 60))
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
