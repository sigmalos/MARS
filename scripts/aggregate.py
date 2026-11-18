"""Aggregate formal-experiment results: mean +- std over seeds and paired tests against a reference configuration.

Usage: python scripts/aggregate.py --results results --out analysis_out/summary.json
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
p.add_argument('--out', default='analysis_out/summary.json')
p.add_argument('--ref', default='base')
args = p.parse_args()

METRICS = ['IC', 'ICIR', 'RIC', 'RICIR', 'MSE', 'MAE']
ORDER = ['base', 'ind', 'corr', 'fine', 'ind_corr', 'multi', 'rand', 'snap', 'sw1', 'sw3', 'multi_jt']
runs = defaultdict(dict)  # (universe, tag) -> seed -> record
for f in glob.glob(os.path.join(args.results, '*.json')):
    r = json.load(open(f))
    if 'stop' not in r:
        continue
    runs[(r['universe'], r['tag'])][r['seed']] = r


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    t = stats.ttest_rel(a, b).pvalue if len(d) > 1 else np.nan
    w = stats.wilcoxon(a, b).pvalue if len(d) > 5 and np.any(d != 0) else np.nan
    return {'diff': float(d.mean()), 'diff_std': float(d.std(ddof=1)) if len(d) > 1 else np.nan,
            'wins': int((d > 0).sum()), 'n': int(len(d)), 'p_t': float(t), 'p_wilcoxon': float(w)}


summary = {}
for (uni, tag), seeds in sorted(runs.items(), key=lambda x: (x[0][0], ORDER.index(x[0][1]) if x[0][1] in ORDER else 99)):
    for proto in ['stop', 'best']:
        key = f'{uni}/{proto}/{tag}'
        vals = {m: np.array([seeds[s][proto]['test'][m] for s in sorted(seeds)]) for m in METRICS}
        ent = {'seeds': sorted(seeds), 'n_params': next(iter(seeds.values()))['n_params'],
               'epoch': [seeds[s][proto]['epoch'] for s in sorted(seeds)]}
        ent.update({m: {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0} for m, v in vals.items()})
        ref = runs.get((uni, args.ref), {})
        common = [s for s in sorted(seeds) if s in ref]
        if tag != args.ref and len(common) > 1:
            ent['vs_ref'] = {m: paired([seeds[s][proto]['test'][m] for s in common], [ref[s][proto]['test'][m] for s in common])
                             for m in METRICS}
        summary[key] = ent

os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
json.dump(summary, open(args.out, 'w'), indent=1)
for uni in ['csi300', 'csi800']:
    for proto in ['stop', 'best']:
        print(f'\n## {uni} / {proto}')
        print('| tag | n | epoch | IC | ICIR | RankIC | RankICIR | dIC [w/n] p_t p_w | dRIC [w/n] p_t p_w |')
        for key, e in summary.items():
            u, pr, tag = key.split('/')
            if u != uni or pr != proto:
                continue
            row = f"| {tag} | {len(e['seeds'])} | {np.mean(e['epoch']):.1f} | " + ' | '.join(
                f"{e[m]['mean']:.4f}±{e[m]['std']:.4f}" if m in ('IC', 'RIC') else f"{e[m]['mean']:.3f}±{e[m]['std']:.3f}"
                for m in ['IC', 'ICIR', 'RIC', 'RICIR'])
            for m in ['IC', 'RIC']:
                v = e.get('vs_ref', {}).get(m)
                row += f" | {v['diff']:+.4f} [{v['wins']}/{v['n']}] {v['p_t']:.3f} {v['p_wilcoxon']:.3f}" if v else ' | -'
            print(row + ' |')
