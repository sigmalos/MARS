"""All numbers used in docs/main.tex: writes analysis_out/final.json and LaTeX table fragments to --tex_dir.

Usage: python scripts/final_analysis.py --tex_dir ../../MASTER/docs/tables
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

p = argparse.ArgumentParser()
p.add_argument('--results', default='results')
p.add_argument('--analysis', default='analysis_out')
p.add_argument('--stage2', default='results_stage2')
p.add_argument('--tex_dir', default='tables')
args = p.parse_args()
os.makedirs(args.tex_dir, exist_ok=True)

NAMES = {'base': 'MASTER', 'ind': '+行业图（CSRC）', 'corr': '+动态相关图', 'fine': '+申万二级图', 'ind_corr': '+行业+相关',
         'multi': '+多关系（三图）', 'rand': '+随机图（对照）', 'snap': '+行业快照图（回填）', 'sw1': '+申万一级图',
         'sw3': '+申万三级图', 'multi_jt': '+多关系+联合 t'}
MAIN = ['base', 'ind', 'corr', 'fine', 'ind_corr', 'multi', 'rand']
CSI300_EXTRA = ['snap', 'sw1', 'sw3']
METRICS = ['IC', 'ICIR', 'RIC', 'RICIR', 'MSE', 'MAE']

runs = defaultdict(dict)
for f in glob.glob(os.path.join(args.results, '*.json')):
    r = json.load(open(f))
    if 'stop' in r:
        runs[(r['universe'], r['tag'])][r['seed']] = r
bt = json.load(open(os.path.join(args.analysis, 'backtest.json')))
unc = json.load(open(os.path.join(args.analysis, 'uncertainty.json')))
out = {}


def vec(uni, tag, proto, metric):
    s = runs[(uni, tag)]
    return np.array([s[k][proto]['test'][metric] for k in sorted(s)])


def paired(a, b):
    d = a - b
    return {'diff': float(d.mean()), 'wins': int((d > 0).sum()), 'n': len(d), 'p_t': float(stats.ttest_rel(a, b).pvalue),
            'p_w': float(stats.wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0,
            'ci': float(stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))}


def holm(ps):
    ps = np.asarray(ps)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(run, 1.0)
    return adj


def fmt(m, s, digits=4):
    return f'{m:.{digits}f}$\\pm${s:.{digits}f}'


def star(pv):
    return '$^{**}$' if pv < 0.01 else ('$^{*}$' if pv < 0.05 else '')


# ------------------------------------------------------------- main table (stop protocol), both universes
for proto in ['stop', 'best']:
    for uni in ['csi300', 'csi800']:
        tags = MAIN + (CSI300_EXTRA if uni == 'csi300' else []) + ['multi_jt']
        rows, raw_p = [], []
        for tag in tags:
            ent = {'epoch': float(np.mean([runs[(uni, tag)][s][proto]['epoch'] for s in runs[(uni, tag)]])),
                   'n_params': next(iter(runs[(uni, tag)].values()))['n_params']}
            for m in METRICS:
                v = vec(uni, tag, proto, m)
                ent[m] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1))}
                if tag != 'base':
                    ent[f'd_{m}'] = paired(v, vec(uni, 'base', proto, m))
            rows.append((tag, ent))
            if tag != 'base':
                raw_p += [ent['d_IC']['p_t'], ent['d_RIC']['p_t']]
        adj = holm(raw_p)
        k = 0
        for tag, ent in rows:
            if tag != 'base':
                ent['d_IC']['p_holm'], ent['d_RIC']['p_holm'] = float(adj[k]), float(adj[k + 1])
                k += 2
        out[f'{uni}/{proto}'] = dict(rows)

        lines = []
        for tag, e in rows:
            cells = [NAMES[tag], f"{e['epoch']:.1f}"]
            for m, dg in [('IC', 4), ('ICIR', 3), ('RIC', 4), ('RICIR', 3)]:
                cells.append(fmt(e[m]['mean'], e[m]['std'], dg))
            for m in ['IC', 'RIC']:
                if tag == 'base':
                    cells.append('--')
                else:
                    d = e[f'd_{m}']
                    cells.append(f"{d['diff']:+.4f} [{d['wins']}/{d['n']}] {d['p_t']:.3f}{star(d['p_holm'])}")
            lines.append(' & '.join(cells) + r' \\')
            if tag in ('rand', 'sw3') and tag != tags[-1]:
                lines.append(r'\midrule')
        open(os.path.join(args.tex_dir, f'main_{uni}_{proto}.tex'), 'w').write('\n'.join(lines) + '\n')

# ------------------------------------------------------------- multi vs single-relation / joint vs two-stage
cmp = {}
for uni in ['csi300', 'csi800']:
    for a, b in [('multi', 'ind'), ('multi', 'ind_corr'), ('ind_corr', 'ind'), ('ind', 'rand'), ('multi', 'rand'), ('ind', 'snap'),
                 ('multi_jt', 'multi'), ('sw1', 'ind'), ('sw1', 'fine')]:
        if (uni, a) in runs and (uni, b) in runs:
            cmp[f'{uni}/{a}-{b}'] = {m: paired(vec(uni, a, 'stop', m), vec(uni, b, 'stop', m)) for m in ['IC', 'RIC', 'ICIR', 'RICIR']}
out['comparisons'] = cmp

# ------------------------------------------------------------- loss-matched comparison (removes the stop-epoch confound)
lm = {}
for uni in ['csi300', 'csi800']:
    for L in [0.98, 0.97, 0.96]:
        for tag in ['base', 'ind', 'corr', 'fine', 'ind_corr', 'multi', 'rand']:
            ic, ric, ep = [], [], []
            for s, r in sorted(runs[(uni, tag)].items()):
                h = r['history']
                e = next((x for x in h if x['train_loss'] <= L), None)
                if e:
                    ic.append(e['test_IC']); ric.append(e['test_RIC']); ep.append(e['epoch'])
            lm[f'{uni}/{L}/{tag}'] = {'IC': float(np.mean(ic)), 'RIC': float(np.mean(ric)), 'epoch': float(np.mean(ep)), 'n': len(ic),
                                      'IC_list': ic, 'RIC_list': ric}
        for tag in ['ind', 'corr', 'fine', 'ind_corr', 'multi', 'rand']:
            a, b = lm[f'{uni}/{L}/{tag}'], lm[f'{uni}/{L}/base']
            if a['n'] == b['n'] == 10:
                for m in ['IC', 'RIC']:
                    a[f'd_{m}'] = paired(np.array(a[f'{m}_list']), np.array(b[f'{m}_list']))
out['loss_matched'] = lm
# matched epoch: test metrics averaged over epochs 10..(min stop epoch over all configs and seeds)
me = {}
for uni in ['csi300', 'csi800']:
    tags = ['base', 'ind', 'corr', 'fine', 'ind_corr', 'multi', 'rand']
    last = min(len(r['history']) for t in tags for r in runs[(uni, t)].values())
    for tag in tags:
        M = {m: np.array([[h[f'test_{m}'] for h in r['history'][10:last]] for _, r in sorted(runs[(uni, tag)].items())]).mean(1)
             for m in ['IC', 'RIC']}
        me[f'{uni}/{tag}'] = {'range': [10, last - 1], **{m: float(M[m].mean()) for m in M}, 'lists': {m: M[m].tolist() for m in M}}
    for tag in tags[1:]:
        for m in ['IC', 'RIC']:
            me[f'{uni}/{tag}'][f'd_{m}'] = paired(np.array(me[f'{uni}/{tag}']['lists'][m]), np.array(me[f'{uni}/base']['lists'][m]))
out['matched_epoch'] = me

# ------------------------------------------------------------- backtests
btab = {}
for uni in ['csi300', 'csi800']:
    tags = MAIN + (CSI300_EXTRA if uni == 'csi300' else []) + ['multi_jt']
    for tag in tags:
        recs = [bt[f'{tag}_{uni}_{s}_stop'] for s in range(10) if f'{tag}_{uni}_{s}_stop' in bt]
        ent = {k: {'mean': float(np.mean([r[k] for r in recs])), 'std': float(np.std([r[k] for r in recs], ddof=1))}
               for k in ['nocost_AR', 'nocost_IR', 'nocost_MDD', 'cost_AR', 'cost_IR', 'cost_MDD', 'turnover']}
        if tag != 'base':
            for k in ['nocost_AR', 'nocost_IR', 'cost_AR', 'cost_IR']:
                a = np.array([r[k] for r in recs])
                b = np.array([bt[f'base_{uni}_{s}_stop'][k] for s in range(10)])
                ent[f'd_{k}'] = paired(a, b)
        btab[f'{uni}/{tag}'] = ent
    lines = []
    for tag in tags:
        e = btab[f'{uni}/{tag}']
        cells = [NAMES[tag], fmt(e['nocost_AR']['mean'], e['nocost_AR']['std'], 3), fmt(e['nocost_IR']['mean'], e['nocost_IR']['std'], 2),
                 f"{e['nocost_MDD']['mean']:.3f}", fmt(e['cost_AR']['mean'], e['cost_AR']['std'], 3), fmt(e['cost_IR']['mean'], e['cost_IR']['std'], 2)]
        cells.append('--' if tag == 'base' else f"{e['d_nocost_IR']['diff']:+.2f} [{e['d_nocost_IR']['wins']}/10] {e['d_nocost_IR']['p_t']:.3f}")
        lines.append(' & '.join(cells) + r' \\')
        if tag in ('rand', 'sw3') and tag != tags[-1]:
            lines.append(r'\midrule')
    open(os.path.join(args.tex_dir, f'backtest_{uni}.tex'), 'w').write('\n'.join(lines) + '\n')
out['backtest'] = btab

# ------------------------------------------------------------- uncertainty: joint vs two-stage, heads
UK = ['NLL', 'ECE', 'PICP50', 'PICP90', 'PICP95', 'PICP99', 'MPIW90', 'spearman_sd_abserr', 'partial_sd_abserr_given_hv',
      'R2_hv', 'R2_hv_sd', 'sd_cv', 'nu']
utab = {}
for uni in ['csi300', 'csi800']:
    s2 = unc['stage2'][uni]
    for head in ['const_gaussian', 'hv_gaussian', 'repr_gaussian', 'const_studentt', 'hv_studentt', 'repr_studentt']:
        utab[f'{uni}/{head}'] = {k: {'mean': float(np.mean([s2[s][head][k] for s in s2 if k in s2[s][head]])),
                                     'std': float(np.std([s2[s][head][k] for s in s2 if k in s2[s][head]], ddof=1))}
                                 for k in UK if k in s2[next(iter(s2))][head]}
        utab[f'{uni}/{head}']['calibration'] = {lv: float(np.mean([s2[s][head]['calibration'][lv] for s in s2]))
                                                for lv in s2[next(iter(s2))][head]['calibration']}
    j = unc['joint'][uni]
    utab[f'{uni}/joint_studentt'] = {k: {'mean': float(np.mean([j[s][k] for s in j])), 'std': float(np.std([j[s][k] for s in j], ddof=1))}
                                     for k in UK if k in j[next(iter(j))]}
    utab[f'{uni}/joint_studentt']['calibration'] = {lv: float(np.mean([j[s]['calibration'][lv] for s in j])) for lv in j[next(iter(j))]['calibration']}
    # paired: repr_studentt vs hv_studentt NLL; repr vs const
    for a, b in [('repr_studentt', 'hv_studentt'), ('repr_studentt', 'const_studentt'), ('repr_studentt', 'repr_gaussian')]:
        utab[f'{uni}/{a}-{b}'] = paired(np.array([s2[s][a]['NLL'] for s in sorted(s2)]), np.array([s2[s][b]['NLL'] for s in sorted(s2)]))
out['uncertainty'] = utab

sz = {}
for uni in ['csi300', 'csi800']:
    S = unc['sizing'][uni]
    for c in ['nocost', 'cost']:
        for w in ['EW', 'inv_sd', 'inv_hv', 'inv_sd2']:
            g = {k: np.array([S[s][w][c][k] for s in sorted(S)]) for k in ['excess_AR', 'excess_IR', 'excess_vol', 'excess_MDD', 'abs_vol', 'abs_MDD', 'turnover']}
            e = {k: {'mean': float(v.mean()), 'std': float(v.std(ddof=1))} for k, v in g.items()}
            if w != 'EW':
                for k in ['excess_IR', 'excess_vol', 'excess_MDD', 'abs_vol']:
                    e[f'd_{k}'] = paired(g[k], np.array([S[s]['EW'][c][k] for s in sorted(S)]))
            sz[f'{uni}/{c}/{w}'] = e
out['sizing'] = sz

# relation attention
ra = {}
for uni in ['csi300', 'csi800']:
    A = [np.load(f)['rel_alpha'].mean(0) for f in sorted(glob.glob(os.path.join(args.results, f'multi_{uni}_*_stop_pred.npz')))]
    ra[uni] = {'mean': np.mean(A, 0).tolist(), 'std': np.std(A, 0, ddof=1).tolist()}
out['rel_alpha'] = ra

json.dump(out, open(os.path.join(args.analysis, 'final.json'), 'w'), indent=1, default=float)
print('ok')
