"""Qlib backtest (official MASTER config) for every saved test prediction file.

TopkDropoutStrategy(topk=30, n_drop=30), deal price close, benchmark SH000300 (csi300) / SH000906 (csi800),
2020-07-01 .. 2022-12-31. Reports excess return without cost (as the official table) and with qlib's default
cost of qlib.contrib.evaluate.backtest_daily (buy 0.05%, sell 0.15%, min 5 CNY; no price-limit check): annualised return AR, information ratio IR, max drawdown MDD.
Usage: python scripts/run_backtests.py --results results --out analysis_out/backtest.json --procs 16
"""
import argparse
import glob
import json
import os
import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--results', default='results')
p.add_argument('--out', default='analysis_out/backtest.json')
p.add_argument('--qlib_dir', default='/data/dataset/qlib_cn_chenditc/extracted/qlib_bin')
p.add_argument('--procs', type=int, default=16)
p.add_argument('--pattern', default='*_pred.npz')
args = p.parse_args()
BENCH = {'csi300': 'SH000300', 'csi800': 'SH000906'}


def init():
    import qlib
    import logging
    qlib.init(provider_uri=args.qlib_dir, region='cn', logging_level=logging.ERROR)


def one(path):
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    name = os.path.basename(path)[:-len('_pred.npz')]
    m = re.match(r'(.+)_(csi300|csi800)_(\d+)_(stop|best|eval)$', name)
    tag, uni, seed, proto = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    ix = np.load(os.path.join(os.path.dirname(path), f'index_{uni}_test.npz'))
    pred = np.load(path)['pred']
    sig = pd.Series(pred, index=pd.MultiIndex.from_arrays([pd.DatetimeIndex(ix['datetime']), ix['instrument']],
                                                          names=['datetime', 'instrument'])).sort_index()
    strategy = TopkDropoutStrategy(signal=sig, topk=30, n_drop=30)
    rep, _ = backtest_daily(start_time='2020-07-01', end_time='2022-12-31', strategy=strategy, account=100000000,
                            benchmark=BENCH[uni], exchange_kwargs={'deal_price': 'close', 'freq': 'day'})
    out = {'tag': tag, 'universe': uni, 'seed': seed, 'proto': proto}
    for key, r in [('nocost', rep['return'] - rep['bench']), ('cost', rep['return'] - rep['bench'] - rep['cost']),
                   ('abs', rep['return'] - rep['cost'])]:
        ra = risk_analysis(r, freq='day')['risk']
        out.update({f'{key}_AR': float(ra['annualized_return']), f'{key}_IR': float(ra['information_ratio']),
                    f'{key}_MDD': float(ra['max_drawdown']), f'{key}_vol': float(ra['std'] * np.sqrt(238))})
    out['turnover'] = float(rep['turnover'].mean())
    return name, out


if __name__ == '__main__':
    done = json.load(open(args.out)) if os.path.exists(args.out) else {}
    files = [f for f in sorted(glob.glob(os.path.join(args.results, args.pattern)))
             if os.path.basename(f)[:-len('_pred.npz')] not in done]
    print(len(done), 'cached,', len(files), 'to run', flush=True)
    with Pool(args.procs, initializer=init) as pool:
        for i, (name, res) in enumerate(pool.imap_unordered(one, files)):
            done[name] = res
            if i % 20 == 0:
                os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
                json.dump(done, open(args.out, 'w'), indent=1)
    json.dump(done, open(args.out, 'w'), indent=1)
    print('DONE', len(done), flush=True)
