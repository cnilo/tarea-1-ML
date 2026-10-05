# -*- coding: utf-8 -*-
"""Datos, calendario, particiones y métricas compartidas por todos los experimentos de la Tarea 2.

Nada de este módulo es específico de un metro: todo lo que depende del dataset (rutas, intervalos por día,
fecha de inicio, tipos de día, feriados, partición, ventana y horizonte) se lee de t2/datasets/<nombre>.json.

Protocolo (motivado por los hallazgos de la Tarea 1):
  * partición cronológica por días en train / validación / test;
  * la escala de normalización se calcula sólo con train (sin fuga desde test);
  * el checkpoint se elige por MAE en validación, nunca por pérdida de entrenamiento ni por test;
  * la referencia estacional es el mismo intervalo del último día anterior del mismo tipo.
"""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEEKDAYS = ['lun', 'mar', 'mie', 'jue', 'vie', 'sab', 'dom']
WEEKDAY_LABELS = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']


def _holidays(spec, dates):
    if not spec:
        return set()
    if isinstance(spec, str) and spec.startswith('auto:'):
        try:
            import holidays
        except ImportError as e:
            raise SystemExit('feriados "auto:XX" requiere el paquete holidays: pip install holidays') from e
        cal = holidays.country_holidays(spec.split(':', 1)[1], years={d.year for d in dates})
        return {d for d in dates if d in cal}
    return {dt.date.fromisoformat(s) for s in spec}


class Dataset:
    """Flujo, adyacencia y calendario de un metro, descritos por t2/datasets/<nombre>.json."""

    def __init__(self, name):
        path = os.path.join(HERE, 'datasets', f'{name}.json')
        if not os.path.exists(path):
            raise SystemExit(f'no existe {path}; ver t2/datasets/santiago.plantilla.json')
        self.name = name
        self.cfg = cfg = json.load(open(path))
        self.title = cfg['nombre']
        self.day = int(cfg['intervalos_por_dia'])
        self.seq = int(cfg['ventana'])
        self.pre = int(cfg['horizonte'])
        self.minutes = int(cfg.get('minutos_por_intervalo', 15))
        self.peak_pct = float(cfg.get('percentil_peak', 90))
        self.reference = cfg.get('referencia_externa')

        flow = pd.read_csv(os.path.join(ROOT, cfg['flujo']), header=0 if cfg.get('cabecera') else None)
        self.stations = [str(c) for c in flow.columns]
        self.flow = flow.values.astype(np.float32)
        self.adj = pd.read_csv(os.path.join(ROOT, cfg['adyacencia']), header=None).values.astype(np.float32)
        T, N = self.flow.shape
        if np.isnan(self.flow).any():
            raise SystemExit(f'{cfg["flujo"]}: hay valores faltantes; complételos (por ejemplo con 0 si no hubo servicio)')
        if T % self.day:
            raise SystemExit(f'{T} filas no es múltiplo de {self.day} intervalos por día: se esperan días completos')
        if self.adj.shape != (N, N):
            raise SystemExit(f'la adyacencia es {self.adj.shape} y el flujo tiene {N} estaciones')
        self.n_days = T // self.day

        start = dt.date.fromisoformat(cfg['fecha_inicio'])
        self.dates = [start + dt.timedelta(days=d) for d in range(self.n_days)]
        self.weekday = [d.weekday() for d in self.dates]
        holi = _holidays(cfg.get('feriados'), self.dates)
        by_weekday = {}
        for typ, days in cfg['tipos_de_dia'].items():
            for w in days:
                by_weekday[WEEKDAYS.index(w)] = typ
        if sorted(by_weekday) != list(range(7)):
            raise SystemExit('tipos_de_dia debe asignar los siete días de la semana')
        self.types = list(cfg['tipos_de_dia'])
        self.day_type = [cfg['tipo_feriado'] if d in holi else by_weekday[w]
                         for d, w in zip(self.dates, self.weekday)]

        part = cfg['particion_dias']
        a, b, c = part['train'], part['train'] + part['val'], part['train'] + part['val'] + part['test']
        if c > self.n_days:
            raise SystemExit(f'la partición suma {c} días y el dataset tiene {self.n_days}')
        self.segments = {'train': (0, a * self.day), 'val': (a * self.day, b * self.day),
                         'test': (b * self.day, c * self.day)}

    # ---------- calendario ----------
    def day_label(self, d):
        return f'{WEEKDAY_LABELS[self.weekday[d]]} {self.dates[d].day}'

    def reference_index(self):
        """Para cada fila tau, la fila del mismo intervalo en el último día anterior del mismo tipo; -1 si no hay."""
        T = self.flow.shape[0]
        ref = np.full(T, -1)
        for tau in range(T):
            d, k = divmod(tau, self.day)
            for dp in range(d - 1, -1, -1):
                if self.day_type[dp] == self.day_type[d]:
                    ref[tau] = dp * self.day + k
                    break
        return ref

    # ---------- ventanas ----------
    def windows(self):
        """Para el instante t (último paso observado):
             x    = flujo en t-ventana+1..t                         (N, ventana)
             xs   = referencia estacional de esos mismos pasos       (N, ventana)
             y    = flujo en t+1..t+horizonte                       (N, horizonte)
             s    = referencia estacional de los objetivos          (N, horizonte) -> ingenuo por tipo de día
             ayer = mismo intervalo del día anterior                (N, horizonte) -> ingenuo 'ayer'
           Una ventana pertenece a la partición donde caen sus objetivos. Sólo se usan instantes con
           referencia disponible para todos los pasos, para que todos los métodos vean las mismas ventanas."""
        flow, D, S, P = self.flow, self.day, self.seq, self.pre
        T = flow.shape[0]
        ref = self.reference_index()
        ok = [t for t in range(S - 1, T - P) if (ref[t - S + 1:t + P + 1] >= 0).all() and t + 1 - D >= 0]
        out = {}
        for name, (lo, hi) in self.segments.items():
            ts = np.array([t for t in ok if t + 1 >= lo and t + P < hi])
            if len(ts) == 0:
                raise SystemExit(f'la partición {name} quedó sin ventanas con referencia disponible')
            win = lambda a, b: np.stack([flow[t + a:t + b].T for t in ts])
            refw = lambda a, b: np.stack([flow[ref[t + a:t + b]].T for t in ts])
            out[name] = dict(t=ts, x=win(-S + 1, 1), xs=refw(-S + 1, 1), y=win(1, P + 1), s=refw(1, P + 1),
                             ayer=win(1 - D, P + 1 - D), last=np.repeat(flow[ts][:, :, None], P, axis=2))
        return out

    def train_scale(self):
        lo, hi = self.segments['train']
        return float(self.flow[lo:hi].max())

    def historical_average(self, ts, by_type=False):
        """Promedio, sobre los días de train, del flujo en el mismo intervalo del día
           (opcionalmente sólo sobre días del mismo tipo que el día objetivo)."""
        lo, hi = self.segments['train']
        rows = np.arange(lo, hi)
        slot, typ = rows % self.day, np.array([self.day_type[r // self.day] for r in rows])
        table = {}
        for k in range(self.day):
            for tp in self.types:
                m = (slot == k) & ((typ == tp) if by_type else True)
                table[(k, tp)] = self.flow[rows[m]].mean(0) if m.any() else self.flow[rows[slot == k]].mean(0)
        return np.stack([np.stack([table[(tau % self.day, self.day_type[tau // self.day])]
                                   for tau in range(t + 1, t + 1 + self.pre)]).T for t in ts])

    def peak_threshold(self, windows):
        """Umbral de 'peak': percentil configurado (90 por defecto) de los objetivos de train."""
        return float(np.percentile(windows['train']['y'], self.peak_pct))


# ---------- métricas ----------
def masked_mape(y, p):
    m = y != 0
    return float(np.mean(np.abs(p[m] - y[m]) / y[m]) * 100)


def metrics(y, p, peak_thr=None):
    """y, p: (S, N, horizonte) en pasajeros por intervalo."""
    e = np.abs(p - y)
    out = {'all': dict(MAE=float(e.mean()), RMSE=float(np.sqrt(((p - y) ** 2).mean())), MAPE=masked_mape(y, p))}
    for h in range(y.shape[-1]):
        eh = e[..., h]
        out[f'h{h + 1}'] = dict(MAE=float(eh.mean()), RMSE=float(np.sqrt((eh ** 2).mean())),
                                MAPE=masked_mape(y[..., h], p[..., h]))
    if peak_thr is not None:
        hi = y > peak_thr
        out['peak'] = dict(MAE=float(e[hi].mean()), share=float(hi.mean()))
        out['offpeak'] = dict(MAE=float(e[~hi].mean()))
    return out


def block_bootstrap_diff(err_a, err_b, block=12, n_boot=2000, seed=0):
    """IC 95 % de MAE(a) - MAE(b) con bootstrap por bloques móviles sobre las ventanas de test
       (las ventanas consecutivas están correlacionadas). err_*: (S, N, horizonte) errores absolutos."""
    d = (err_a - err_b).reshape(err_a.shape[0], -1).mean(1)
    S = len(d)
    rng = np.random.default_rng(seed)
    starts = np.arange(S - block + 1)
    k = int(np.ceil(S / block))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, k)])[:S]
        boots[i] = d[idx].mean()
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def add_dataset_arg(ap):
    ap.add_argument('--dataset', default='hangzhou', help='nombre de t2/datasets/<nombre>.json')
    return ap
