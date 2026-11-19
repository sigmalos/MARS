"""Global daily-return matrix + per-sample indices, so that correlations over any look-back window can be computed
on the fly (GPU) inside the model.

  <out>/returns.npz            ret [n_cal, n_inst] float32 close-to-close returns (NaN = missing), cal (datetime64[D]),
                               instrument (U12)
  <out>/<uni>_<split>_idx.npz  cal_idx int32 (row of the sample date in ret), inst_idx int32 (column of the stock),
                               instrument / datetime for the alignment check (same order as ds.get_index())
A sample on day t may only use rows [cal_idx - W, cal_idx) i.e. returns strictly before t.
"""
import argparse
import pickle

import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument('--data_dir', default='/data/dataset/MASTER/opensource')
p.add_argument('--qlib_dir', default='/data/dataset/qlib_cn_chenditc/extracted/qlib_bin')
p.add_argument('--out', default='/data/dataset/MASTER_relations')
p.add_argument('--start', default='2006-01-01')
args = p.parse_args()

datasets = {}
for u in ['csi300', 'csi800']:
    for split in ['train', 'valid', 'test']:
        idx = pickle.load(open(f'{args.data_dir}/{u}_dl_{split}.pkl', 'rb')).get_index()
        datasets[(u, split)] = (np.asarray(idx.get_level_values('instrument')).astype('U12'),
                                idx.get_level_values('datetime').values.astype('datetime64[D]'))
all_inst = sorted(set(np.concatenate([v[0] for v in datasets.values()])))

import logging
import qlib
from qlib.data import D
qlib.init(provider_uri=args.qlib_dir, region='cn', logging_level=logging.ERROR)
cal = pd.DatetimeIndex(D.calendar(start_time=args.start, end_time='2022-12-31'))
px = D.features(all_inst, ['$close'], start_time=args.start, end_time='2022-12-31')['$close'].unstack('instrument')
px = px.reindex(index=cal, columns=all_inst)
ret = (px / px.shift(1) - 1).values.astype(np.float32)
cal_d = cal.values.astype('datetime64[D]')
np.savez(f'{args.out}/returns.npz', ret=ret, cal=cal_d, instrument=np.array(all_inst, dtype='U12'))
pos = {d: i for i, d in enumerate(cal_d)}
col = {s: i for i, s in enumerate(all_inst)}
for (u, split), (inst, dt) in datasets.items():
    ci = np.array([pos[d] for d in dt], np.int32)
    ii = np.array([col[s] for s in inst], np.int32)
    assert ci.min() >= 250, f'need >= 250 return rows before the first sample ({u} {split})'
    np.savez(f'{args.out}/{u}_{split}_idx.npz', cal_idx=ci, inst_idx=ii, instrument=inst, datetime=dt)
    print(u, split, len(dt), 'first cal row', ci.min(), flush=True)
print('ret', ret.shape, 'missing %.1f%%' % (100 * np.isnan(ret).mean()))
