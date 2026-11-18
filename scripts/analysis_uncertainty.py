"""Uncertainty analysis: (1) joint vs two-stage probabilistic quality, (2) uncertainty-based position sizing.

Position sizing (top-30 by mu, daily rebalancing, close-to-close, validated backtester `backtest.py`):
  EW        equal weight (official setting)
  inv_sd    weight ~ 1 / sd_pred   (two-stage Student-t predictive std)
  inv_hv    weight ~ 1 / hv20      (control: naive inverse historical volatility)
  inv_sd2   weight ~ 1 / sd_pred^2
Costs: buy 0.05 %, sell 0.15 % (same as qlib backtest_daily used for the main backtests) and without cost.
Usage: python scripts/analysis_uncertainty.py --out analysis_out/uncertainty.json
"""
import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from uncertainty import prob_metrics, std_of  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument('--results', default='results')
p.add_argument('--stage2', default='results_stage2')
p.add_argument('--relation_dir', default='/data/dataset/MASTER_relations')
p.add_argument('--qlib_dir', default='/data/dataset/qlib_cn_chenditc/extracted/qlib_bin')
p.add_argument('--out', default='analysis_out/uncertainty.json')
p.add_argument('--procs', type=int, default=10)
args = p.parse_args()
BENCH = {'csi300': 'SH000300', 'csi800': 'SH000906'}


def index(uni):
    ix = np.load(os.path.join(args.results, f'index_{uni}_test.npz'))
    return pd.MultiIndex.from_arrays([ix['instrument'], pd.DatetimeIndex(ix['datetime'])], names=['instrument', 'datetime'])


def joint_vs_twostage(uni):
    """Probabilistic metrics of the jointly trained Student-t model vs two-stage heads on the same seeds."""
    hv = np.load(f'{args.relation_dir}/{uni}_test.npz')['hv20']
    idx = index(uni)
    day = pd.factorize(idx.get_level_values('datetime'))[0]
    out = {}
    for f in sorted(glob.glob(os.path.join(args.results, f'multi_jt_{uni}_*_stop_pred.npz'))):
        seed = int(os.path.basename(f).split('_')[3])
        z = np.load(f)
        out[seed] = prob_metrics(z['pred'], z['scale'], z['label_z'], nu=z['nu'], hv=hv, day=day)
    return out


_prices = {}


def bt_one(task):
    import backtest as bt
    uni, seed = task
    if uni not in _prices:
        import logging
        import qlib
        qlib.init(provider_uri=args.qlib_dir, region='cn', logging_level=logging.ERROR)
        _prices[uni] = bt.PriceData(args.qlib_dir, index(uni).get_level_values(0).unique(), benchmark=BENCH[uni])
    prices, idx = _prices[uni], index(uni)
    z = np.load(os.path.join(args.stage2, f'multi_{uni}_{seed}_stop_stage2.npz'))
    sd = std_of(z['scale_studentt'], z['nu_studentt'])
    hv = np.where(np.isfinite(z['hv20']) & (z['hv20'] > 0), z['hv20'], np.nan)
    hv = pd.Series(hv, index=idx).groupby(level='datetime').transform(lambda s: s.fillna(s.median())).values
    score = pd.Series(z['mu'], index=idx)
    res = {}
    for name, w in [('EW', None), ('inv_sd', 1 / sd), ('inv_hv', 1 / hv), ('inv_sd2', 1 / sd ** 2)]:
        ws = None if w is None else pd.Series(w, index=idx)
        _, m0 = bt.topk_backtest(score, prices, weight=ws)
        _, m1 = bt.topk_backtest(score, prices, weight=ws, buy_cost=0.0005, sell_cost=0.0015)
        res[name] = {'nocost': m0, 'cost': m1}
    return uni, seed, res


if __name__ == '__main__':
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    result = {'joint': {u: joint_vs_twostage(u) for u in ['csi300', 'csi800']}, 'stage2': {}, 'sizing': {}}
    for f in sorted(glob.glob(os.path.join(args.stage2, 'multi_*_stop_stage2.json'))):
        r = json.load(open(f))
        result['stage2'].setdefault(r['universe'], {})[r['seed']] = r
    tasks = [(u, s) for u in result['stage2'] for s in sorted(result['stage2'][u])]
    with Pool(args.procs) as pool:
        for uni, seed, res in pool.imap_unordered(bt_one, tasks):
            result['sizing'].setdefault(uni, {})[seed] = res
            print('sizing', uni, seed, {k: round(v['cost']['excess_IR'], 3) for k, v in res.items()}, flush=True)
    json.dump(result, open(args.out, 'w'), indent=1, default=float)
    print('written', args.out)
