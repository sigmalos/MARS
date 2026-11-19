"""Attach pre-computed stock-relation arrays (see scripts/build_relations.py) to a qlib TSDataSampler.

Each file <relation_dir>/<universe>_<split>.npz stores per-sample arrays aligned with ds.get_index():
  csrc, csrc_snap, csrc_rand, sw1, sw2, sw3 : int32 industry ids (-1 = unknown -> isolated node)
  corr10                                    : int16 [n, 10] positions (within the same day) of the top-10
                                              correlated stocks over the previous 60 trading days (-1 = none)
  hv20                                      : float32 20-day historical volatility of daily returns (up to t)
  instrument, datetime                      : alignment check
With returns=True, <universe>_<split>_idx.npz (cal_idx, inst_idx) is attached as well and the global return
matrix (scripts/build_returns.py) is stored as ds.ret_matrix, so that correlations over any window can be computed
on the fly (rows [cal_idx - W, cal_idx) = returns strictly before the sample date).
"""
import numpy as np


_RET = {}


def _check(z, ds):
    index = ds.get_index()
    inst = np.asarray(index.get_level_values('instrument')).astype('U12')
    dt = index.get_level_values('datetime').values.astype('datetime64[D]')
    assert len(z['instrument']) == len(ds), 'relation file does not match dataset length'
    assert (z['instrument'] == inst).all() and (z['datetime'] == dt).all(), 'relation file not aligned with dataset'


def attach(ds, relation_dir, universe, split, keys=None, returns=False):
    z = np.load(f'{relation_dir}/{universe}_{split}.npz', allow_pickle=False)
    _check(z, ds)
    keys = keys if keys is not None else [k for k in z.files if k not in ('instrument', 'datetime')]
    ds.relation_arrays = {k: z[k] for k in keys}
    if returns:
        zi = np.load(f'{relation_dir}/{universe}_{split}_idx.npz', allow_pickle=False)
        _check(zi, ds)
        ds.relation_arrays.update(cal_idx=zi['cal_idx'], inst_idx=zi['inst_idx'])
        if relation_dir not in _RET:
            _RET[relation_dir] = np.load(f'{relation_dir}/returns.npz')['ret']
        ds.ret_matrix = _RET[relation_dir]
    return ds.relation_arrays
