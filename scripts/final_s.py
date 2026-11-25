"""Round-3 test-set evaluation. Every rule parameter comes from analysis_out/selection.json (inner split);
the test set is only read here, once.

For each full-protocol configuration (base, ic_w60, ic_wstar, final_g, final_gt) x universe x seed:
  test metrics under the stopping rules
    off    official: first epoch with train MSE <= 0.95
    vbest  best official-validation IC epoch (67 days; formal-experiment protocol, for reference)
    epoch  fixed epoch E* of the configuration (inner split)
    loss   per-model threshold L* (inner split)
    ens    mean z-scored prediction of the K* epochs ending at E* (inner split)
  Qlib backtests (TopkDropout top-30, close, benchmark index, backtest_daily default costs) for the rules off and
  sel (= the rule family chosen on the inner split) with n_drop = 30 (official) and n_drop* (inner split).
Writes analysis_out/final_s.json.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import traj as T  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--results', default='results')
p.add_argument('--out', default='analysis_out')
p.add_argument('--qlib_dir', default='/data/dataset/qlib_cn_chenditc/extracted/qlib_bin')
p.add_argument('--procs', type=int, default=16)
args = p.parse_args()
SEL = json.load(open(os.path.join(args.out, 'selection.json')))
TEST = ('2020-07-01', '2022-12-31')
RULES = ['off', 'vbest', 'epoch', 'loss', 'ens']
runs = T.load_runs(args.results, 'full')


def rule_epochs(uni, tag, r):
    """epoch (or (end, K) for the ensemble) selected by each rule for one run."""
    st = SEL[uni]['stats'][SEL['_inner_of'][uni].get(tag, tag)]
    loss = [h['train_loss'] for h in r['history']]
    return {'off': T.first_below(loss, T.OFFICIAL), 'vbest': int(np.argmax([h['valid_IC'] for h in r['history']])),
            'epoch': st['E_star'], 'loss': T.first_below(loss, st['L_star']), 'ens': (st['E_star'], st['K_star'])}


def rule_pred(P, groups, sel_e):
    if isinstance(sel_e, tuple):
        return T.ensemble(P, groups, *sel_e)
    return P[sel_e].astype(np.float32)


out = {'metrics': {}, 'epochs': {}}
for (uni, tag), seeds in sorted(runs.items()):
    inst, dt, y = T.index(args.results, uni, 'test')
    groups = T.day_groups(dt)
    for s, r in sorted(seeds.items()):
        P = T.preds(args.results, r, 'test')
        E = rule_epochs(uni, tag, r)
        for rule in RULES:
            out['metrics'][f'{uni}/{tag}/{rule}/{s}'] = T.daily_ic(rule_pred(P, groups, E[rule]), y, groups)
        out['epochs'][f'{uni}/{tag}/{s}'] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in E.items()}
        out['metrics'][f'{uni}/{tag}/curve/{s}'] = {'IC': [h['test_IC'] for h in r['history']], 'RIC': [h['test_RIC'] for h in r['history']],
                                                    'loss': [h['train_loss'] for h in r['history']]}
    print(uni, tag, 'done', flush=True)


# ---------------------------------------------------------------- backtests
def init():
    import logging
    import qlib
    qlib.init(provider_uri=args.qlib_dir, region='cn', logging_level=logging.ERROR)


def bt(task):
    from mars import backtest
    uni, tag, s, rule, nd = task
    r = runs[(uni, tag)][s]
    inst, dt, y = T.index(args.results, uni, 'test')
    groups = T.day_groups(dt)
    E = rule_epochs(uni, tag, r)
    pred = rule_pred(T.preds(args.results, r, 'test'), groups, E[rule])
    score = pd.Series(pred, index=pd.MultiIndex.from_arrays([pd.DatetimeIndex(dt), inst], names=['datetime', 'instrument']))
    return task, backtest.qlib_backtest(score, uni, TEST[0], TEST[1], n_drop=nd)


tasks = []
for (uni, tag), seeds in runs.items():
    for s in seeds:
        for rule in sorted({'off', SEL[uni]['rule']}):
            for nd in sorted({30, SEL[uni]['n_drop_star']}):
                tasks.append((uni, tag, s, rule, nd))
out['backtest'] = {}
with Pool(args.procs, initializer=init) as pool:
    for i, (task, res) in enumerate(pool.imap_unordered(bt, tasks)):
        out['backtest']['/'.join(map(str, task))] = res
        if i % 50 == 0:
            print('backtest', i, '/', len(tasks), flush=True)


# ---------------------------------------------------------------- summaries and paired tests
def paired(a, b):
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    return {'diff': float(d.mean()), 'wins': int((d > 0).sum()), 'n': len(d), 'p_t': float(stats.ttest_rel(a, b).pvalue),
            'p_w': float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0,
            'ci': float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))}


def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order, adj, run = np.argsort(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def vec(uni, tag, rule, m):
    return np.array([out['metrics'][f'{uni}/{tag}/{rule}/{s}'][m] for s in sorted(runs[(uni, tag)])])


def bvec(uni, tag, rule, nd, k):
    return np.array([out['backtest'][f'{uni}/{tag}/{s}/{rule}/{nd}'][k] for s in sorted(runs[(uni, tag)])])


summ = {}
for uni in ['csi300', 'csi800']:
    tags = [t for t in ['base', 'ic_w60', 'ic_wstar', 'final_g', 'final_gt'] if (uni, t) in runs]
    srule, nds = SEL[uni]['rule'], SEL[uni]['n_drop_star']
    S = {}
    for tag in tags:
        for rule in RULES:
            S[f'{tag}/{rule}'] = {m: {'mean': float(vec(uni, tag, rule, m).mean()), 'std': float(vec(uni, tag, rule, m).std(ddof=1))}
                                  for m in ['IC', 'ICIR', 'RIC', 'RICIR']}
        for rule in sorted({'off', srule}):
            for nd in sorted({30, nds}):
                S[f'{tag}/{rule}/bt{nd}'] = {k: {'mean': float(bvec(uni, tag, rule, nd, k).mean()), 'std': float(bvec(uni, tag, rule, nd, k).std(ddof=1))}
                                             for k in ['nocost_AR', 'nocost_IR', 'nocost_MDD', 'cost_AR', 'cost_IR', 'cost_MDD', 'turnover']}
    # (a) stopping rules: each rule vs official for the same model; graph vs base under each rule
    tests = {}
    for tag in [t for t in ['base', 'ic_w60'] if t in tags]:
        for rule in ['vbest', 'epoch', 'loss', 'ens']:
            for m in ['IC', 'RIC']:
                tests[f'rule/{tag}/{rule}-off/{m}'] = paired(vec(uni, tag, rule, m), vec(uni, tag, 'off', m))
    for rule in RULES:
        for m in ['IC', 'RIC', 'ICIR', 'RICIR']:
            tests[f'graph/{rule}/ic_w60-base/{m}'] = paired(vec(uni, 'ic_w60', rule, m), vec(uni, 'base', rule, m))
    # (b) step-wise improvements under the selected rule, and every configuration vs base / official
    chain = [('ic_wstar', 'ic_w60'), ('final_g', 'ic_wstar'), ('final_gt', 'final_g')]
    for a, b in chain:
        a_, b_ = SEL['_alias'].get(uni, {}).get(a, a), SEL['_alias'].get(uni, {}).get(b, b)
        if a_ in tags and b_ in tags and a_ != b_:
            for m in ['IC', 'RIC', 'ICIR', 'RICIR']:
                tests[f'step/{a}-{b}/{m}'] = paired(vec(uni, a_, srule, m), vec(uni, b_, srule, m))
    for tag in tags:
        if tag != 'base':
            for m in ['IC', 'RIC']:
                tests[f'vsbase/{tag}/{m}'] = paired(vec(uni, tag, srule, m), vec(uni, 'base', srule, m))
                tests[f'vsofficial/{tag}/{m}'] = paired(vec(uni, tag, srule, m), vec(uni, 'base', 'off', m))
    # (c) portfolio: cost IR of each configuration (selected rule) vs base, n_drop* vs 30, penalty
    for tag in tags:
        for nd in sorted({30, nds}):
            if tag != 'base':
                for k in ['cost_IR', 'nocost_IR', 'turnover']:
                    tests[f'bt/{tag}-base/{srule}/{nd}/{k}'] = paired(bvec(uni, tag, srule, nd, k), bvec(uni, 'base', srule, nd, k))
        if nds != 30:
            for k in ['cost_IR', 'nocost_IR', 'turnover', 'cost_AR']:
                tests[f'bt/{tag}/ndrop{nds}-30/{k}'] = paired(bvec(uni, tag, srule, nds, k), bvec(uni, tag, srule, 30, k))
        for k in ['cost_IR', 'nocost_IR']:
            tests[f'bt/{tag}/{srule}-off/30/{k}'] = paired(bvec(uni, tag, srule, 30, k), bvec(uni, tag, 'off', 30, k)) if srule != 'off' else None
    # Holm over the main family: every configuration vs base under the selected rule (IC, RIC)
    fam = [k for k in tests if k.startswith('vsbase/')]
    for k, a in zip(fam, holm([tests[k]['p_t'] for k in fam])):
        tests[k]['p_holm'] = float(a)
    summ[uni] = {'S': S, 'tests': {k: v for k, v in tests.items() if v is not None}, 'tags': tags, 'rule': srule, 'n_drop_star': nds}
out['summary'] = summ
json.dump(out, open(os.path.join(args.out, 'final_s.json'), 'w'), indent=1, default=float)
for uni in summ:
    print('==', uni, 'rule', summ[uni]['rule'], 'n_drop*', summ[uni]['n_drop_star'])
    for k, v in summ[uni]['S'].items():
        if '/bt' not in k:
            print(f"  {k:22s} IC {v['IC']['mean']:.4f}±{v['IC']['std']:.4f} RIC {v['RIC']['mean']:.4f} ICIR {v['ICIR']['mean']:.3f}")
        else:
            print(f"  {k:22s} AR {v['nocost_AR']['mean']:.3f} IR {v['nocost_IR']['mean']:.2f} | cost AR {v['cost_AR']['mean']:.3f} IR {v['cost_IR']['mean']:.2f} | turnover {v['turnover']['mean']:.3f}")
    for k, v in summ[uni]['tests'].items():
        print(f"  {k:40s} {v['diff']:+.4f} [{v['wins']}/{v['n']}] p={v['p_t']:.3f}" + (f" holm={v['p_holm']:.3f}" if 'p_holm' in v else ''))
