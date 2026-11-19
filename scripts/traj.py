"""Shared helpers for round-3 analysis: load trajectories, apply stopping rules, epoch ensembles, daily IC."""
import glob
import json
import os

import numpy as np
import pandas as pd

SMOOTH = 3                                      # centred moving average (epochs) for the fixed-epoch rule
LOSS_GRID = [round(0.99 - 0.005 * i, 3) for i in range(17)]   # 0.99 .. 0.91
ENS_K = [1, 3, 5, 9]
OFFICIAL = 0.95


def load_runs(result_dir, protocol):
    runs = {}
    for f in glob.glob(os.path.join(result_dir, f'*_{protocol}_*.json')):
        r = json.load(open(f))
        runs.setdefault((r['universe'], r['tag']), {})[r['seed']] = r
    return runs


def preds(result_dir, r, set_name):
    return np.load(os.path.join(result_dir, f"{r['tag']}_{r['universe']}_{r['protocol']}_{r['seed']}_preds.npz"))[set_name]


def index(result_dir, uni, set_name):
    z = np.load(os.path.join(result_dir, f'index_{uni}_{set_name}.npz'))
    return z['instrument'], z['datetime'], z['label_z']


def day_groups(dt):
    days, starts = np.unique(dt, return_index=True)
    return list(zip(starts, list(starts[1:]) + [len(dt)]))


def cs_z(p, groups):
    z = np.empty_like(p, dtype=np.float32)
    for a, b in groups:
        x = p[a:b].astype(np.float32)
        z[a:b] = (x - x.mean()) / (x.std() + 1e-12)
    return z


def daily_ic(p, y, groups):
    from scipy.stats import rankdata
    ic, ric = [], []
    for a, b in groups:
        m = ~np.isnan(y[a:b])
        pp, yy = p[a:b][m].astype(np.float64), y[a:b][m]
        ic.append(np.corrcoef(pp, yy)[0, 1])
        ric.append(np.corrcoef(rankdata(pp), rankdata(yy))[0, 1])
    ic, ric = np.array(ic), np.array(ric)
    return {'IC': float(np.nanmean(ic)), 'ICIR': float(np.nanmean(ic) / np.nanstd(ic)),
            'RIC': float(np.nanmean(ric)), 'RICIR': float(np.nanmean(ric) / np.nanstd(ric))}


def first_below(losses, L):
    for e, l in enumerate(losses):
        if l <= L:
            return e
    return len(losses) - 1


def smooth(curve, w=SMOOTH):
    c = np.asarray(curve, dtype=float)
    out = np.empty_like(c)
    h = w // 2
    for i in range(len(c)):
        out[i] = c[max(0, i - h):i + h + 1].mean()
    return out


def ensemble(P, groups, e_end, K):
    """mean of the cross-sectionally z-scored predictions of epochs e_end-K+1 .. e_end."""
    lo = max(0, e_end - K + 1)
    return np.mean([cs_z(P[e], groups) for e in range(lo, e_end + 1)], axis=0)
