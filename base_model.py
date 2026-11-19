import os
import time
import copy
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import torch
import torch.optim as optim


def zscore(x):
    return (x - x.mean()).div(x.std())


def drop_extreme(x):
    sorted_tensor, indices = x.sort()
    N = x.shape[0]
    percent_2_5 = int(0.025*N)
    # Exclude top 2.5% and bottom 2.5% values
    filtered_indices = indices[percent_2_5:-percent_2_5]
    mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
    mask[filtered_indices] = True
    return mask, x[mask]


def daily_ic(pred, label):
    """Pearson / Spearman correlation of one cross-section, NaN labels ignored (same as pandas .corr)."""
    m = ~np.isnan(label)
    p, l = pred[m], label[m]
    if len(l) < 2:
        return np.nan, np.nan
    ic = np.corrcoef(p, l)[0, 1]
    ric = np.corrcoef(rankdata(p), rankdata(l))[0, 1]
    return ic, ric


def cs_zscore_np(label):
    """Cross-sectional z-score over the non-NaN labels of one day (NaN kept)."""
    z = np.full_like(label, np.nan)
    v = ~np.isnan(label)
    if v.sum() > 1:
        z[v] = (label[v] - label[v].mean()) / label[v].std(ddof=1)
    return z


def _ffill_rows(a):
    """Row-wise forward fill of NaN (vectorised version of qlib.data.dataset.np_ffill)."""
    mask = np.isnan(a)
    idx = np.where(~mask, np.arange(a.shape[1])[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return np.take_along_axis(a, idx, axis=1)


class DayBatches:
    """All samples of one trading day form one batch (as in the official DailyBatchSamplerRandom).

    The [N, T, F] tensor of a day is gathered with pre-computed index arrays instead of calling
    TSDataSampler.__getitem__ sample by sample; the result is bit-identical (checked in tests) but
    several times faster. Relation arrays attached to the dataset (see relations.py) are sliced per day.
    `date_range` = (first, last) (inclusive, strings or None) restricts the batches to a contiguous range of days;
    outputs of `evaluate` are then indexed compactly over the selected samples (`index()`).
    """

    def __init__(self, ds, date_range=None):
        fillna_type = getattr(ds, 'fillna_type', 'none')
        assert fillna_type in ('none', 'ffill', 'ffill+bfill')
        index = ds.get_index()
        self.ds = ds
        self.instruments = np.asarray(index.get_level_values('instrument'))
        dates = index.get_level_values('datetime')
        assert (np.diff(dates.values.astype('int64')) >= 0).all(), 'index must be sorted by datetime'
        counts = pd.Series(index=index, dtype=float).groupby('datetime').size()
        starts = np.r_[0, np.cumsum(counts.values)[:-1]]
        sel = np.ones(len(counts), bool)
        if date_range is not None:
            lo, hi = date_range
            if lo is not None:
                sel &= counts.index >= pd.Timestamp(lo)
            if hi is not None:
                sel &= counts.index <= pd.Timestamp(hi)
        self.day_dates = counts.index[sel]
        self.counts = counts.values[sel]
        self.starts = starts[sel]                       # global row of the first sample of each selected day
        self.out_starts = np.r_[0, np.cumsum(self.counts)[:-1]]   # compact row in evaluate() outputs
        self.n = int(self.counts.sum())
        rc = np.asarray(ds.idx_map)
        if rc.dtype == object:
            rc = np.array([tuple(x) for x in rc])
        rows, cols = rc[:, 0].astype(np.int64), rc[:, 1].astype(np.int64)
        win = rows[:, None] + np.arange(-ds.step_len + 1, 1)[None, :]
        ind = ds.idx_arr[np.clip(win, 0, None), cols[:, None]].astype(np.float64)
        ind[win < 0] = np.nan
        if fillna_type == 'ffill':
            ind = _ffill_rows(ind)
        elif fillna_type == 'ffill+bfill':
            ind = _ffill_rows(_ffill_rows(ind)[:, ::-1])[:, ::-1]
        self.sample_idx = np.nan_to_num(ind, nan=ds.nan_idx).astype(np.int64)  # [n_samples, T]
        self.relations = getattr(ds, 'relation_arrays', {})
        self.ret_matrix = getattr(ds, 'ret_matrix', None)

    def __len__(self):
        return len(self.counts)

    def rows(self):
        return np.concatenate([np.arange(s, s + n) for s, n in zip(self.starts, self.counts)])

    def index(self):
        return self.ds.get_index()[self.rows()]

    def order(self, shuffle):
        index = np.arange(len(self.counts))
        if shuffle:
            np.random.shuffle(index)  # same RNG usage as the official sampler
        return index

    def get(self, k):
        s, n = self.starts[k], self.counts[k]
        data = self.ds.data_arr[self.sample_idx[s:s + n]]  # [N, T, F]
        rel = {name: arr[s:s + n] for name, arr in self.relations.items()}
        return torch.from_numpy(data), {'inst': self.instruments[s:s + n], 'date': self.day_dates[k], 'rel': rel,
                                        'slice': (self.out_starts[k], n), 'ret': self.ret_matrix}


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path='model/', save_prefix='',
                 drop_extreme_label=True, select_metric='IC', turnover_lambda=0.0):
        self.n_epochs = n_epochs
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred
        # True for opensource training data; False for original training data (already processed when dumped).
        self.drop_extreme_label = drop_extreme_label
        self.select_metric = select_metric
        # training-time turnover penalty: lambda * mean((z(pred_t) - z(pred_{t-1}))^2) over stocks present on both days,
        # pred_{t-1} from an eval-mode forward without gradient (target), z = cross-sectional z-score
        self.turnover_lambda = turnover_lambda

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
        self.fitted = -1

        self.model = None
        self.train_optimizer = None

        self.save_path = save_path
        self.save_prefix = save_prefix
        self._batch_cache = {}

    def init_model(self):
        if self.model is None:
            raise ValueError("model has not been initialized")

        self.train_optimizer = optim.Adam(self.model.parameters(), self.lr)
        self.model.to(self.device)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        loss = (pred[mask]-label[mask])**2
        return torch.mean(loss)

    # ------------------------------------------------------------------
    # Hooks that improved variants override.
    # ------------------------------------------------------------------
    def _forward(self, feature, info, keep=None, return_repr=False):
        """Run the network on one day. `keep` (bool mask over the day's samples) is set when extreme labels
        were dropped during training. Returns a dict with at least key 'pred' (N,)."""
        return {'pred': self.model(feature)}

    def _train_loss(self, out, label):
        """Returns (loss used for backprop, monitored MSE used by the stop criterion)."""
        loss = self.loss_fn(out['pred'], label)
        return loss, loss

    def batches(self, dataset, date_range=None):
        key = (id(dataset), date_range)
        if key not in self._batch_cache:
            self._batch_cache[key] = DayBatches(dataset, date_range)
        return self._batch_cache[key]

    # ------------------------------------------------------------------
    def _turnover_penalty(self, db, k, info, keep, pred):
        pdata, pinfo = db.get(k - 1)
        self.model.eval()
        with torch.no_grad():
            prev = self._forward(pdata[:, :, 0:-1].to(self.device).float(), pinfo, None)['pred']
        self.model.train()
        inst = info['inst'] if keep is None else info['inst'][keep.cpu().numpy()]
        _, ia, ib = np.intersect1d(inst, pinfo['inst'], assume_unique=True, return_indices=True)
        if len(ia) < 10:
            return None
        p = pred[torch.from_numpy(ia).to(self.device)]
        q = prev[torch.from_numpy(ib).to(self.device)]
        p = (p - p.mean()) / (p.std() + 1e-8)
        q = (q - q.mean()) / (q.std() + 1e-8)
        return ((p - q) ** 2).mean()

    def train_epoch(self, dataset, date_range=None):
        self.model.train()
        losses, to_losses = [], []
        db = self.batches(dataset, date_range)
        for k in db.order(shuffle=True):
            data, info = db.get(k)
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            keep = None
            # Additional process on labels (opensource data): drop extreme labels and CSZScoreNorm.
            if self.drop_extreme_label:
                keep, label = drop_extreme(label)
                feature = feature[keep, :, :]
                label = zscore(label)

            out = self._forward(feature.float(), info, keep)
            loss, monitor = self._train_loss(out, label)
            losses.append(monitor.item())
            if self.turnover_lambda > 0 and k > 0:
                to = self._turnover_penalty(db, k, info, keep, out['pred'])
                if to is not None:
                    loss = loss + self.turnover_lambda * to
                    to_losses.append(to.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        self.last_turnover_loss = float(np.mean(to_losses)) if to_losses else float('nan')
        return float(np.mean(losses))

    def evaluate(self, dataset, return_repr=False, date_range=None):
        """Predict every sample of `dataset` (all stocks, NaN labels only ignored in the metrics).
        Returns (metrics, outputs) where outputs maps name -> array aligned with dataset.get_index()
        (or with DayBatches(dataset, date_range).index() if a date range is given)."""
        db = self.batches(dataset, date_range)
        n = db.n
        outputs, labels = {}, np.full(n, np.nan, dtype=np.float32)
        ic, ric = [], []
        self.model.eval()
        with torch.no_grad():
            for k in db.order(shuffle=False):
                data, info = db.get(k)
                s, cnt = info['slice']
                feature = data[:, :, 0:-1].to(self.device)
                label = data[:, -1, -1].numpy()
                out = self._forward(feature.float(), info, None, return_repr=return_repr)
                for name, v in out.items():
                    v = v.detach().cpu().numpy()
                    if v.ndim >= 1 and v.shape[0] == cnt:
                        if name not in outputs:
                            outputs[name] = np.zeros((n,) + v.shape[1:], dtype=np.float32)
                        outputs[name][s:s + cnt] = v
                labels[s:s + cnt] = cs_zscore_np(label)
                a, b = daily_ic(out['pred'].detach().cpu().numpy(), label)
                ic.append(a)
                ric.append(b)
        ic, ric = np.array(ic), np.array(ric)
        self.last_daily_ic = ic
        v = ~np.isnan(labels)
        err = outputs['pred'][v] - labels[v]
        metrics = {
            'IC': float(np.nanmean(ic)), 'ICIR': float(np.nanmean(ic) / np.nanstd(ic)),
            'RIC': float(np.nanmean(ric)), 'RICIR': float(np.nanmean(ric) / np.nanstd(ric)),
            'MSE': float(np.mean(err ** 2)), 'MAE': float(np.mean(np.abs(err))),
        }
        outputs['label_z'] = labels
        return metrics, outputs

    def predict(self, dl_test):
        metrics, outputs = self.evaluate(dl_test)
        return pd.Series(outputs['pred'], index=dl_test.get_index()), metrics

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device, weights_only=True))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None, dl_test=None, full_epochs=True, keep_outputs=('pred',), patience=None):
        """Train for n_epochs. Two model-selection protocols are recorded from the same trajectory:
          * 'stop': official MASTER criterion - first epoch with train loss <= train_stop_loss_thred
                    (last epoch if never reached). With full_epochs=False training ends there, exactly as the
                    official code; with full_epochs=True training continues, which does not change the
                    trajectory up to that epoch (evaluation consumes no randomness).
          * 'best': epoch with the best validation `select_metric`. With `patience`, training ends once the
                    official criterion has been reached and the validation metric has not improved for
                    `patience` epochs (standard early stopping; at most n_epochs).
        dl_test is only *evaluated* (stored for the selected epochs); it never influences training/selection."""
        os.makedirs(self.save_path, exist_ok=True)
        self.history, self.selected = [], {}
        best_val = -np.inf
        for step in range(self.n_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(dl_train)
            self.fitted = step
            rec = {'epoch': step, 'train_loss': train_loss}
            vm, _ = self.evaluate(dl_valid)
            rec.update({f'valid_{k}': v for k, v in vm.items()})
            test_out = None
            if dl_test is not None:
                tm, test_out = self.evaluate(dl_test)
                rec.update({f'test_{k}': v for k, v in tm.items()})
            rec['time'] = time.time() - t0
            self.history.append(rec)
            print("Epoch %d, train_loss %.6f, valid ic %.4f, rankic %.4f | test ic %.4f, rankic %.4f, %.0fs." % (
                step, train_loss, vm['IC'], vm['RIC'], rec.get('test_IC', np.nan), rec.get('test_RIC', np.nan), rec['time']), flush=True)

            def snapshot():
                return {'epoch': step, 'state': copy.deepcopy(self.model.state_dict()),
                        'outputs': {k: v.copy() for k, v in (test_out or {}).items() if k in keep_outputs or k == 'label_z'},
                        'test': {k[5:]: v for k, v in rec.items() if k.startswith('test_')},
                        'valid': {k[6:]: v for k, v in rec.items() if k.startswith('valid_')}}
            if vm[self.select_metric] > best_val:
                best_val = vm[self.select_metric]
                self.selected['best'] = snapshot()
            reached = train_loss <= self.train_stop_loss_thred
            if 'stop' not in self.selected and (reached or step == self.n_epochs - 1):
                self.selected['stop'] = snapshot()
                if not full_epochs:
                    break
            if patience is not None and 'stop' in self.selected and step - self.selected['best']['epoch'] >= patience:
                break
        for name, sel in self.selected.items():
            torch.save(sel['state'], f'{self.save_path}/{self.save_prefix}_{self.seed}_{name}.pkl')
        # leave the model in the official-criterion state
        self.model.load_state_dict(self.selected['stop']['state'])
        self.fitted = self.selected['stop']['epoch']
        return self.selected

    def fit_trajectory(self, train, evals, epochs):
        """Train for exactly `epochs` epochs (no stopping) and record, after every epoch, the train loss, the
        turnover penalty and metrics + float16 predictions on every evaluation set. All stopping / selection rules
        are applied offline to this trajectory. train = (dataset, date_range); evals = {name: (dataset, date_range)}."""
        self.history, self.eval_labels = [], {}
        preds = {name: [] for name in evals}
        for step in range(epochs):
            t0 = time.time()
            train_loss = self.train_epoch(*train)
            self.fitted = step
            rec = {'epoch': step, 'train_loss': train_loss, 'turnover_loss': self.last_turnover_loss}
            for name, (ds, rng) in evals.items():
                m, out = self.evaluate(ds, date_range=rng)
                rec.update({f'{name}_{k}': v for k, v in m.items()})
                preds[name].append(out['pred'].astype(np.float16))
                if step == 0:
                    self.eval_labels[name] = out['label_z']
            rec['time'] = time.time() - t0
            self.history.append(rec)
            print('Epoch %d, train_loss %.6f, to %.4f | ' % (step, train_loss, self.last_turnover_loss) + ', '.join(
                '%s ic %.4f ric %.4f' % (n, rec[f'{n}_IC'], rec[f'{n}_RIC']) for n in evals) + ', %.0fs' % rec['time'], flush=True)
        return self.history, {k: np.stack(v) for k, v in preds.items()}
