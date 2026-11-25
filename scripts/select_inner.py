"""Hyper-parameter and stopping-rule selection on the inner split (no test data is read).

Pre-registered procedure (per universe):
  1. For every inner configuration and every stopping rule compute the inner-validation IC (mean over seeds):
       off   official: first epoch with train MSE <= 0.95
       epoch fixed epoch E* = argmax of the 3-epoch moving average of the seed-mean inner-val IC curve
       loss  per-model threshold L* in 0.99..0.91: first epoch with train MSE <= L*
       ens   mean of z-scored predictions of the K* epochs ending at E*, K* in {1,3,5,9}
  2. Rule family = the rule with the highest inner-val IC averaged over all configurations.
  3. Under that rule: window W* (ic_w20..ic_w250), edge variant (ic_w60 and e_*: must beat the unweighted ic_w60),
     turnover lambda* (ic_w60 and to_*) by inner-val Qlib cost-adjusted IR, portfolio n_drop* (TopkDropout, top-30)
     by inner-val cost-adjusted IR of ic_w60.
Outputs analysis_out/selection.json and analysis_out/fullB_configs.json (input of make_jobs_s.py fullB).
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import traj as T  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--results', default='results')
p.add_argument('--out', default='analysis_out')
p.add_argument('--qlib_dir', default='/data/dataset/qlib_cn_chenditc/extracted/qlib_bin')
p.add_argument('--procs', type=int, default=16)
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)
WINDOWS = {'ic_w20': 20, 'ic_w40': 40, 'ic_w60': 60, 'ic_w120': 120, 'ic_w250': 250}
EDGES = ['ic_w60', 'e_corrw', 'e_indw', 'e_inds', 'e_corrs', 'e_both']
LAMBDAS = {'ic_w60': 0.0, 'to_0.1': 0.1, 'to_0.3': 0.3, 'to_1.0': 1.0}
NDROP = [30, 10, 5, 3]
IVAL = ('2018-01-02', '2020-03-31')

runs = T.load_runs(args.results, 'inner')
sel = {}
for uni in ['csi300', 'csi800']:
    inst, dt, y = T.index(args.results, uni, 'ival')
    groups = T.day_groups(dt)
    stats = {}
    for (u, tag), seeds in sorted(runs.items()):
        if u != uni:
            continue
        S = sorted(seeds)
        ic = np.array([[h['ival_IC'] for h in seeds[s]['history']] for s in S])
        ric = np.array([[h['ival_RIC'] for h in seeds[s]['history']] for s in S])
        loss = np.array([[h['train_loss'] for h in seeds[s]['history']] for s in S])
        sm = T.smooth(ic.mean(0))
        E = int(np.argmax(sm))
        e_off = [T.first_below(l, T.OFFICIAL) for l in loss]
        L_val = {L: float(np.mean([ic[i, T.first_below(loss[i], L)] for i in range(len(S))])) for L in T.LOSS_GRID}
        L_star = max(L_val, key=L_val.get)
        P = {s: T.preds(args.results, seeds[s], 'ival') for s in S}
        ens = {}
        for K in T.ENS_K:
            vals = [T.daily_ic(T.ensemble(P[s], groups, E, K), y, groups) for s in S]
            ens[K] = {m: float(np.mean([v[m] for v in vals])) for m in ['IC', 'RIC', 'ICIR', 'RICIR']}
        K_star = max(ens, key=lambda k: ens[k]['IC'])
        stats[tag] = {
            'seeds': S, 'n_params': seeds[S[0]]['n_params'], 'E_star': E, 'L_star': L_star, 'K_star': K_star,
            'off': {'epoch': float(np.mean(e_off)), 'IC': float(np.mean([ic[i, e] for i, e in enumerate(e_off)])),
                    'RIC': float(np.mean([ric[i, e] for i, e in enumerate(e_off)]))},
            'epoch': {'epoch': E, 'IC': float(ic[:, E].mean()), 'IC_smooth': float(sm[E]), 'RIC': float(ric[:, E].mean())},
            'loss': {'L': L_star, 'epoch': float(np.mean([T.first_below(l, L_star) for l in loss])), 'IC': L_val[L_star],
                     'RIC': float(np.mean([ric[i, T.first_below(loss[i], L_star)] for i in range(len(S))]))},
            'ens': {'K': K_star, **ens[K_star]}, 'ens_all': ens,
            'curve_IC': ic.mean(0).tolist(), 'curve_RIC': ric.mean(0).tolist(), 'curve_loss': loss.mean(0).tolist(),
        }
        print(f"{uni} {tag:8s} off ep{stats[tag]['off']['epoch']:.1f} IC {stats[tag]['off']['IC']:.4f} | E*={E} IC {ic[:, E].mean():.4f}"
              f" | L*={L_star} IC {L_val[L_star]:.4f} | ens K*={K_star} IC {ens[K_star]['IC']:.4f} RIC {ens[K_star]['RIC']:.4f}", flush=True)
    rules = ['off', 'epoch', 'loss', 'ens']
    rule_mean = {r: float(np.mean([stats[t][r]['IC'] for t in stats])) for r in rules}
    rule = max(rule_mean, key=rule_mean.get)
    score = lambda t: stats[t][rule]['IC']
    W_tag = max([t for t in WINDOWS if t in stats], key=score)
    E_tag = max([t for t in EDGES if t in stats], key=score)
    sel[uni] = {'stats': stats, 'rule_mean_IC': rule_mean, 'rule': rule, 'W_star': WINDOWS[W_tag], 'W_tag': W_tag,
                'edge_tag': E_tag}
    print(f'{uni}: rule means {rule_mean} -> rule {rule}; W* = {WINDOWS[W_tag]} ({W_tag}); edge = {E_tag}', flush=True)


# ---------------------------------------------------------------- inner-val backtests (lambda, n_drop)
def predictions_for(uni, tag, seed):
    st = sel[uni]['stats'][tag]
    r = runs[(uni, tag)][seed]
    P = T.preds(args.results, r, 'ival')
    inst, dt, y = T.index(args.results, uni, 'ival')
    groups = T.day_groups(dt)
    rule = sel[uni]['rule']
    if rule == 'off':
        e = T.first_below([h['train_loss'] for h in r['history']], T.OFFICIAL); p = P[e].astype(np.float32)
    elif rule == 'epoch':
        p = P[st['E_star']].astype(np.float32)
    elif rule == 'loss':
        e = T.first_below([h['train_loss'] for h in r['history']], st['L_star']); p = P[e].astype(np.float32)
    else:
        p = T.ensemble(P, groups, st['E_star'], st['K_star'])
    return pd.Series(p, index=pd.MultiIndex.from_arrays([pd.DatetimeIndex(dt), inst], names=['datetime', 'instrument']))


def init():
    import logging
    import qlib
    qlib.init(provider_uri=args.qlib_dir, region='cn', logging_level=logging.ERROR)


def bt(task):
    from mars import backtest
    uni, tag, seed, nd = task
    return task, backtest.qlib_backtest(predictions_for(uni, tag, seed), uni, IVAL[0], IVAL[1], n_drop=nd)


tasks = []
for uni in sel:
    for tag in LAMBDAS:
        if tag in sel[uni]['stats']:
            for s in sel[uni]['stats'][tag]['seeds']:
                for nd in (NDROP if tag == 'ic_w60' else [30]):
                    tasks.append((uni, tag, s, nd))
    for s in sel[uni]['stats']['base']['seeds']:
        tasks.append((uni, 'base', s, 30))
res = {}
with Pool(args.procs, initializer=init) as pool:
    for task, out in pool.imap_unordered(bt, tasks):
        res[task] = out
for uni in sel:
    agg = lambda tag, nd, k: float(np.mean([res[(uni, tag, s, nd)][k] for s in sel[uni]['stats'][tag]['seeds']]))
    lam = {LAMBDAS[t]: {k: agg(t, 30, k) for k in ['cost_IR', 'nocost_IR', 'cost_AR', 'turnover']} for t in LAMBDAS if t in sel[uni]['stats']}
    nd = {n: {k: agg('ic_w60', n, k) for k in ['cost_IR', 'nocost_IR', 'cost_AR', 'turnover']} for n in NDROP}
    sel[uni]['inner_backtest'] = {'lambda': lam, 'n_drop': nd, 'base': {k: agg('base', 30, k) for k in ['cost_IR', 'nocost_IR', 'turnover']}}
    sel[uni]['lambda_star'] = max(lam, key=lambda l: lam[l]['cost_IR'])
    sel[uni]['n_drop_star'] = max(nd, key=lambda n: nd[n]['cost_IR'])
    print(uni, 'lambda grid', {l: round(v['cost_IR'], 3) for l, v in lam.items()}, '-> lambda*', sel[uni]['lambda_star'])
    print(uni, 'n_drop grid', {n: (round(v['cost_IR'], 3), round(v['turnover'], 3)) for n, v in nd.items()}, '-> n_drop*', sel[uni]['n_drop_star'])

# ---------------------------------------------------------------- full-protocol phase-B configurations
EDGE_KW = {'ic_w60': {}, 'e_corrw': {'corr_edge': 'weight'}, 'e_indw': {'ind_edge': 'weight'}, 'e_inds': {'ind_edge': 'signed'},
           'e_corrs': {'corr_mode': 'signed', 'corr_edge': 'signed'},
           'e_both': {'ind_edge': 'signed', 'corr_mode': 'signed', 'corr_edge': 'signed'}}
cfg = {'ic_wstar': {}, 'final_g': {}, 'final_gt': {}}
for uni in sel:
    W, ek, lam = sel[uni]['W_star'], EDGE_KW[sel[uni]['edge_tag']], sel[uni]['lambda_star']
    cfg['ic_wstar'][uni] = ['ind,dcorr', {'corr_window': W}, 0.0]
    cfg['final_g'][uni] = ['ind,dcorr', {'corr_window': W, **ek}, 0.0]
    cfg['final_gt'][uni] = ['ind,dcorr', {'corr_window': W, **ek}, lam]
# drop configurations that duplicate an existing full run (ic_w60) or each other
full_b, alias = {}, {}
for uni in sel:
    seen = {json.dumps(['ind,dcorr', {'corr_window': 60}, 0.0], sort_keys=True): 'ic_w60'}
    for tag in ['ic_wstar', 'final_g', 'final_gt']:
        key = json.dumps(cfg[tag][uni], sort_keys=True)
        if key in seen:
            alias.setdefault(uni, {})[tag] = seen[key]
        else:
            seen[key] = tag
            full_b.setdefault(tag, ['', {}, 0.0])[1][uni] = cfg[tag][uni]
# inner runs are also needed for new combinations (their own E*, L*, K* for the stopping rules)
INNER_KEY = {json.dumps(['ind,dcorr', {'corr_window': w}, 0.0], sort_keys=True): t for t, w in WINDOWS.items()}
INNER_KEY.update({json.dumps(['ind,dcorr', {'corr_window': 60, **kw}, 0.0], sort_keys=True): t for t, kw in EDGE_KW.items()})
INNER_KEY.update({json.dumps(['ind,dcorr', {'corr_window': 60}, l], sort_keys=True): t for t, l in LAMBDAS.items()})
inner_of, inner_b = {}, {}
for uni in sel:
    for tag in ['ic_wstar', 'final_g', 'final_gt']:
        key = json.dumps(cfg[tag][uni], sort_keys=True)
        if key in INNER_KEY:
            inner_of.setdefault(uni, {})[tag] = INNER_KEY[key]
        else:
            inner_of.setdefault(uni, {})[tag] = tag
            inner_b.setdefault(tag, ['', {}, 0.0])[1][uni] = cfg[tag][uni]
    inner_of[uni].update({'base': 'base', 'ic_w60': 'ic_w60'})
full_b['_inner'] = inner_b
sel['_configs'], sel['_alias'], sel['_inner_of'] = cfg, alias, inner_of
json.dump(full_b, open(os.path.join(args.out, 'fullB_configs.json'), 'w'), indent=1)
json.dump(sel, open(os.path.join(args.out, 'selection.json'), 'w'), indent=1, default=float)
print('phase-B full runs:', {t: list(v[1]) for t, v in full_b.items() if t != '_inner'}, '| inner runs:', {t: list(v[1]) for t, v in inner_b.items()}, '| aliases:', alias, '| inner_of:', inner_of)
