"""Build per-sample stock-relation arrays for the MASTER datasets (see relations.py for the file format).

Point-in-time rule: for a sample on trading day d only information effective on or before d is used.
  csrc       CSRC industry. baostock monthly snapshots; record effective from its `updateDate`.
  csrc_snap  CSRC industry of the 2026-11-18 snapshot, back-filled (pre-experiment graph, look-ahead!).
  csrc_rand  control: random industry graph. Every stock gets a fixed random key (seed 123); on each day the
             multiset of that day's true csrc labels (incl. unknown) is assigned to the stocks in key order. Group
             sizes / edge counts are identical to csrc every day and memberships are persistent, but random.
  sw1/2/3    Shenwan level 1/2/3 from cninfo change records (old standard, SW2021 from its effective date).
             Days before a stock's first Shenwan record use that first record (only affects early 2008).
  corr10     top-10 Pearson correlation of daily returns over the 60 trading days strictly before d
             (>= 40 valid returns required), positions refer to the same day's cross-section.
  hv20       std of daily returns over the 20 trading days ending at d (>= 10 valid).

Usage: python scripts/build_relations.py --out /data/dataset/MASTER_relations --data_out data
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--data_dir', default='/data/dataset/MASTER/opensource')
p.add_argument('--qlib_dir', default='/data/dataset/qlib_cn_chenditc/extracted/qlib_bin')
p.add_argument('--csrc_raw', default='/data/dataset/MASTER_relations/csrc_pit_raw_*.csv')
p.add_argument('--cninfo_raw', default='/data/dataset/MASTER_relations/cninfo_raw')
p.add_argument('--snapshot', default='data/stock_industry.csv')
p.add_argument('--out', default='/data/dataset/MASTER_relations')
p.add_argument('--data_out', default='data', help='where the processed point-in-time tables are written')
p.add_argument('--k', type=int, default=10)
p.add_argument('--corr_window', type=int, default=60)
args = p.parse_args()

SW_OLD, SW_NEW = '申银万国行业分类标准(旧)', '申银万国行业分类标准'

# ---------------------------------------------------------------- datasets
datasets = {}
for u in ['csi300', 'csi800']:
    for split in ['train', 'valid', 'test']:
        ds = pickle.load(open(f'{args.data_dir}/{u}_dl_{split}.pkl', 'rb'))
        idx = ds.get_index()
        datasets[(u, split)] = (np.asarray(idx.get_level_values('instrument')).astype('U12'),
                                idx.get_level_values('datetime').values.astype('datetime64[D]'))
all_inst = sorted(set(np.concatenate([v[0] for v in datasets.values()])))
print('instruments', len(all_inst), flush=True)


class PITTable:
    """instrument -> sorted (effective_date, label) records; lookup(inst, dates) returns labels effective at dates."""
    def __init__(self, df, date_col, label_col, backfill=False):
        self.rec = {}
        for inst, g in df.sort_values(date_col).groupby('instrument'):
            self.rec[inst] = (g[date_col].values.astype('datetime64[D]'), g[label_col].values)
        self.backfill = backfill

    def lookup(self, inst, dates):
        out = np.full(len(dates), '', dtype=object)
        if inst not in self.rec:
            return out, np.zeros(len(dates), bool)
        eff, lab = self.rec[inst]
        pos = np.searchsorted(eff, dates, side='right') - 1
        ok = pos >= 0
        out[ok] = lab[pos[ok]]
        filled = np.zeros(len(dates), bool)
        if self.backfill:
            filled = ~ok
            out[~ok] = lab[0]
        return out, filled


# ---------------------------------------------------------------- CSRC point-in-time
csrc = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob(args.csrc_raw))], ignore_index=True)
csrc = csrc[csrc['industry'].fillna('') != ''].drop_duplicates(['instrument', 'updateDate'])[['instrument', 'updateDate', 'industry']]
os.makedirs(args.data_out, exist_ok=True)
csrc.sort_values(['instrument', 'updateDate']).to_csv(f'{args.data_out}/csrc_pit.csv', index=False)
csrc_t = PITTable(csrc, 'updateDate', 'industry')
print('CSRC records', len(csrc), 'snapshot dates', csrc.updateDate.nunique(), csrc.updateDate.min(), csrc.updateDate.max(), flush=True)

snap = pd.read_csv(args.snapshot).fillna({'industry': ''})
snap_map = dict(zip(snap['instrument'], snap['industry']))

# ---------------------------------------------------------------- Shenwan point-in-time (cninfo)
rows = []
for f in glob.glob(f'{args.cninfo_raw}/*.json'):
    inst = os.path.basename(f)[:-5]
    for r in json.load(open(f)):
        if r.get('F002V') in (SW_OLD, SW_NEW) and r.get('VARYDATE'):
            rows.append((inst, r['VARYDATE'][:10], r['F002V'], r.get('F003V'), r.get('F004V'), r.get('F005V'), r.get('F006V')))
sw = pd.DataFrame(rows, columns=['instrument', 'varydate', 'standard', 'code', 'l1', 'l2', 'l3']).drop_duplicates()
sw.sort_values(['instrument', 'varydate']).to_csv(f'{args.data_out}/sw_pit.csv', index=False)
print('SW records', len(sw), sw.standard.value_counts().to_dict(), flush=True)
sw_old = {lv: PITTable(sw[sw.standard == SW_OLD], 'varydate', lv, backfill=True) for lv in ['l1', 'l2', 'l3']}
sw_new = {lv: PITTable(sw[sw.standard == SW_NEW], 'varydate', lv) for lv in ['l1', 'l2', 'l3']}

# ---------------------------------------------------------------- prices -> returns
import qlib
from qlib.data import D
qlib.init(provider_uri=args.qlib_dir, region='cn')
cal = pd.DatetimeIndex(D.calendar(start_time='2007-06-01', end_time='2022-12-31'))
px = D.features(all_inst, ['$close'], start_time='2007-06-01', end_time='2022-12-31')['$close'].unstack('instrument')
px = px.reindex(index=cal, columns=all_inst)
ret = (px / px.shift(1) - 1).values.astype(np.float64)  # [n_days, n_inst]
cal_pos = {d: i for i, d in enumerate(cal.values.astype('datetime64[D]'))}
col_of = {s: i for i, s in enumerate(all_inst)}
print('price matrix', ret.shape, 'missing close %.1f%%' % (100 * px.isna().values.mean()), flush=True)

rng = np.random.RandomState(123)
rand_key = dict(zip(all_inst, rng.permutation(len(all_inst))))

vocab = {k: {} for k in ['csrc', 'csrc_snap', 'csrc_rand', 'sw1', 'sw2', 'sw3']}
vocab_shared = {'csrc_rand': 'csrc'}  # random graph uses the same label space as csrc


def encode(key, labels):
    voc = vocab[vocab_shared.get(key, key)]
    out = np.full(len(labels), -1, np.int32)
    for i, s in enumerate(labels):
        if isinstance(s, str) and s != '':
            out[i] = voc.setdefault(s, len(voc))
    return out


stats = {}
for (u, split), (inst, dt) in datasets.items():
    n = len(inst)
    lab = {k: np.full(n, '', dtype=object) for k in vocab}
    sw_backfilled = np.zeros(n, bool)
    order = np.argsort(inst, kind='stable')
    for s in np.unique(inst):
        rows_s = np.where(inst == s)[0]
        d = dt[rows_s]
        lab['csrc'][rows_s] = csrc_t.lookup(s, d)[0]
        lab['csrc_snap'][rows_s] = snap_map.get(s, '')
        for lv, key in [('l1', 'sw1'), ('l2', 'sw2'), ('l3', 'sw3')]:
            old, filled = sw_old[lv].lookup(s, d)
            new, _ = sw_new[lv].lookup(s, d)
            lab[key][rows_s] = np.where(new != '', new, old)
            if lv == 'l1':
                sw_backfilled[rows_s] = filled & (new == '')
    arrays = {k: encode(k, v) for k, v in lab.items() if k != 'csrc_rand'}
    days_, starts_ = np.unique(dt, return_index=True)
    bnd = list(starts_) + [n]
    rnd = np.empty(n, np.int32)
    for j in range(len(days_)):
        a, b = bnd[j], bnd[j + 1]
        keys = np.array([rand_key[x] for x in inst[a:b]])
        seg = np.empty(b - a, np.int32)
        seg[np.argsort(keys, kind='stable')] = np.sort(arrays['csrc'][a:b])
        rnd[a:b] = seg
    arrays['csrc_rand'] = rnd

    # correlation graph and historical volatility, day by day
    corr10 = np.full((n, args.k), -1, np.int16)
    hv20 = np.full(n, np.nan, np.float32)
    days, starts = np.unique(dt, return_index=True)
    bounds = list(starts) + [n]
    nb_counts, same_ind = [], []
    for j, day in enumerate(days):
        a, b = bounds[j], bounds[j + 1]
        cols = np.array([col_of[s] for s in inst[a:b]])
        t = cal_pos[day]
        W = ret[t - args.corr_window:t, cols]               # previous 60 returns (strictly before day)
        H = ret[t - 19:t + 1, cols]                          # 20 returns up to and including day
        hcnt = (~np.isnan(H)).sum(0)
        hv = np.nanstd(H, axis=0, ddof=1) if H.shape[0] > 1 else np.full(len(cols), np.nan)
        hv20[a:b] = np.where(hcnt >= 10, hv, np.nan)
        m = ~np.isnan(W)
        cnt = m.sum(0)
        ok = cnt >= 40
        mean = np.where(m, W, 0).sum(0) / np.maximum(cnt, 1)
        Xc = np.where(m, W - mean, 0.0)
        sd = np.sqrt((Xc ** 2).sum(0) / np.maximum(cnt - 1, 1))
        Z = np.where(ok & (sd > 0), Xc / np.where(sd > 0, sd, 1), 0.0)
        C = Z.T @ Z / (args.corr_window - 1)
        valid = ok & (sd > 0)
        C[:, ~valid] = -np.inf
        np.fill_diagonal(C, -np.inf)
        kk = min(args.k, max(int(valid.sum()) - 1, 0))
        if kk > 0:
            top = np.argpartition(-C, kk - 1, axis=1)[:, :kk]
            top = np.take_along_axis(top, np.argsort(-np.take_along_axis(C, top, 1), axis=1), 1)
            top[~valid] = -1
            corr10[a:b, :kk] = top
        nb_counts.append(valid.mean())
        ind = arrays['csrc'][a:b]
        tv = top[valid] if kk > 0 else np.zeros((0, 1), int)
        if len(tv):
            same_ind.append(np.mean((ind[valid][:, None] == ind[tv]) & (ind[valid][:, None] >= 0)))
    arrays['corr10'], arrays['hv20'] = corr10, hv20

    # statistics
    st = {'samples': int(n), 'days': int(len(days)), 'sw_backfilled_frac': float(sw_backfilled.mean()),
          'corr_valid_frac': float(np.mean(nb_counts)), 'corr_nb_same_csrc_frac': float(np.mean(same_ind)),
          'hv20_valid_frac': float(np.isfinite(hv20).mean())}
    for key in vocab:
        ids = arrays[key]
        g = pd.DataFrame({'d': dt, 'id': ids})
        g = g[g.id >= 0]
        sz = g.groupby(['d', 'id']).size()
        deg = (sz * (sz - 1)).groupby(level=0).sum() / pd.Series(dt).value_counts().sort_index().values
        st[key] = {'coverage': float((ids >= 0).mean()), 'groups_per_day': float(sz.groupby(level=0).size().mean()),
                   'neighbours_per_stock': float(deg.mean())}
    stats[f'{u}_{split}'] = st
    print(u, split, json.dumps(st, ensure_ascii=False), flush=True)
    np.savez_compressed(f'{args.out}/{u}_{split}.npz', instrument=inst, datetime=dt, **arrays)

json.dump({'stats': stats, 'vocab_sizes': {k: len(v) for k, v in vocab.items()}}, open(f'{args.data_out}/relation_stats.json', 'w'),
          indent=1, ensure_ascii=False)
json.dump(vocab, open(f'{args.out}/vocab.json', 'w'), ensure_ascii=False)
print('DONE', flush=True)
