"""Attach pre-computed stock-relation arrays (see scripts/build_relations.py) to a qlib TSDataSampler.

Each file <relation_dir>/<universe>_<split>.npz stores per-sample arrays aligned with ds.get_index():
  csrc, csrc_snap, csrc_rand, sw1, sw2, sw3 : int32 industry ids (-1 = unknown -> isolated node)
  corr10                                    : int16 [n, 10] positions (within the same day) of the top-10
                                              correlated stocks over the previous 60 trading days (-1 = none)
  hv20                                      : float32 20-day historical volatility of daily returns (up to t)
  instrument, datetime                      : alignment check
"""
import numpy as np


def attach(ds, relation_dir, universe, split, keys=None):
    z = np.load(f'{relation_dir}/{universe}_{split}.npz', allow_pickle=False)
    index = ds.get_index()
    inst = np.asarray(index.get_level_values('instrument')).astype('U12')
    dt = index.get_level_values('datetime').values.astype('datetime64[D]')
    assert len(z['instrument']) == len(ds), 'relation file does not match dataset length'
    assert (z['instrument'] == inst).all() and (z['datetime'] == dt).all(), 'relation file not aligned with dataset'
    keys = keys if keys is not None else [k for k in z.files if k not in ('instrument', 'datetime')]
    ds.relation_arrays = {k: z[k] for k in keys}
    return ds.relation_arrays
