"""Vectorised Top-k portfolio backtest that mirrors the official MASTER/Qlib setting.

Official config: TopkDropoutStrategy(topk=30, n_drop=30), deal price = close, benchmark = index.
With n_drop == topk the strategy holds the top-30 stocks of the previous day's signal, i.e.
  signal on day s  ->  trade at the close of day s+1  ->  earn the close-to-close return of day s+2.
Metrics follow qlib.contrib.evaluate.risk_analysis (mode='sum'):
  AR = mean(r) * 252,  IR = mean(r) / std(r) * sqrt(252),  MDD = min(cumsum(r) - cummax(cumsum(r))).
"""
import numpy as np
import pandas as pd

N_YEAR = 252
BENCHMARK = {'csi300': 'SH000300', 'csi800': 'SH000906'}


class PriceData:
    def __init__(self, qlib_dir, instruments, start='2020-06-01', end='2023-01-31', benchmark='SH000300'):
        import qlib
        from qlib.data import D
        qlib.init(provider_uri=qlib_dir, region='cn')
        cal = pd.DatetimeIndex(D.calendar(start_time=start, end_time=end))
        close = D.features(list(instruments), ['$close'], start_time=start, end_time=end)['$close'].unstack('instrument')
        self.close = close.reindex(index=cal, columns=sorted(set(instruments)))
        self.close.columns = [c.upper() for c in self.close.columns]
        b = D.features([benchmark], ['$close'], start_time=start, end_time=end)['$close'].droplevel(0).reindex(cal)
        self.ret = self.close / self.close.shift(1) - 1
        self.bench_ret = b / b.shift(1) - 1
        self.cal = cal


def risk(r):
    r = pd.Series(r).dropna()
    cs = r.cumsum()
    return {'AR': float(r.mean() * N_YEAR), 'vol': float(r.std() * np.sqrt(N_YEAR)),
            'IR': float(r.mean() / r.std() * np.sqrt(N_YEAR)), 'MDD': float((cs - cs.cummax()).min())}


def topk_backtest(score, prices, topk=30, weight=None, buy_cost=0.0, sell_cost=0.0, start='2020-07-01', end='2022-12-31'):
    """score / weight: pd.Series indexed by (instrument, datetime). weight (optional, > 0) gives the relative
    position size inside the top-k (default equal weight). Only return days in [start, end] are kept (official
    backtest window). Returns a daily DataFrame and summary metrics."""
    s = score.unstack('instrument').reindex(columns=prices.close.columns)
    w_in = None if weight is None else weight.unstack('instrument').reindex(index=s.index, columns=prices.close.columns)
    cal = prices.cal
    pos = {d: i for i, d in enumerate(cal)}
    recs, prev_w = [], pd.Series(dtype=float)
    for d in s.index:
        i = pos.get(d)
        if i is None or i + 2 >= len(cal):
            continue
        t1, t2 = cal[i + 1], cal[i + 2]
        if t1 < pd.Timestamp(start) or t2 > pd.Timestamp(end):
            continue
        row = s.loc[d]
        tradable = row.notna() & prices.close.loc[t1].notna()
        top = row[tradable].nlargest(topk).index
        if len(top) == 0:
            continue
        if w_in is None:
            w = pd.Series(1.0 / len(top), index=top)
        else:
            ww = w_in.loc[d, top].clip(lower=0).fillna(0)
            w = ww / ww.sum() if ww.sum() > 0 else pd.Series(1.0 / len(top), index=top)
        r = prices.ret.loc[t2, top].fillna(0.0)
        gross = float((w * r).sum())
        # turnover against the previous (drifted) holdings
        allc = w.index.union(prev_w.index)
        dw = w.reindex(allc, fill_value=0) - prev_w.reindex(allc, fill_value=0)
        cost = buy_cost * dw.clip(lower=0).sum() + sell_cost * (-dw).clip(lower=0).sum()
        recs.append({'datetime': t2, 'port': gross - cost, 'port_gross': gross, 'bench': float(prices.bench_ret.loc[t2]),
                     'turnover': float(dw.abs().sum() / 2), 'cost': float(cost)})
        drift = w * (1 + r)
        prev_w = drift / drift.sum()
    df = pd.DataFrame(recs).set_index('datetime')
    df['excess'] = df['port'] - df['bench']
    df['excess_gross'] = df['port_gross'] - df['bench']
    m = {f'excess_{k}': v for k, v in risk(df['excess']).items()}
    m.update({f'excess_gross_{k}': v for k, v in risk(df['excess_gross']).items()})
    m.update({f'abs_{k}': v for k, v in risk(df['port']).items()})
    m['turnover'] = float(df['turnover'].mean())
    return df, m


def qlib_backtest(score, universe, start, end, topk=30, n_drop=30, benchmark=None):
    """Official MASTER/Qlib backtest (TopkDropoutStrategy, close price, qlib backtest_daily default costs
    buy 0.05% / sell 0.15% / min 5 CNY). score: Series indexed by (datetime, instrument). qlib must be initialised.
    n_drop < topk is qlib's built-in turnover constraint (at most n_drop holdings are replaced per day)."""
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.contrib.evaluate import backtest_daily, risk_analysis
    strategy = TopkDropoutStrategy(signal=score.sort_index(), topk=topk, n_drop=n_drop)
    rep, _ = backtest_daily(start_time=start, end_time=end, strategy=strategy, account=100000000,
                            benchmark=benchmark or BENCHMARK[universe], exchange_kwargs={'deal_price': 'close', 'freq': 'day'})
    out = {}
    for key, r in [('nocost', rep['return'] - rep['bench']), ('cost', rep['return'] - rep['bench'] - rep['cost'])]:
        ra = risk_analysis(r, freq='day')['risk']
        out.update({f'{key}_AR': float(ra['annualized_return']), f'{key}_IR': float(ra['information_ratio']),
                    f'{key}_MDD': float(ra['max_drawdown'])})
    out['turnover'] = float(rep['turnover'].mean())
    return out
