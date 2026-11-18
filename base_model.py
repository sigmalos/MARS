import os
import time
import numpy as np
import pandas as pd
import copy

from torch.utils.data import Sampler
import torch
import torch.optim as optim

def calc_ic(pred, label):
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

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

def drop_na(x):
    mask = ~x.isnan()
    return mask, x[mask]

class DailyBatchSamplerRandom(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        # calculate number of samples in each batch
        self.daily_count = pd.Series(index=self.data_source.get_index()).groupby("datetime").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        if self.shuffle:
            index = np.arange(len(self.daily_count))
            np.random.shuffle(index)
            for i in index:
                yield np.arange(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
        else:
            for idx, count in zip(self.daily_index, self.daily_count):
                yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data_source)


class SequenceModel():
    def __init__(self, n_epochs, lr, GPU=None, seed=None, train_stop_loss_thred=None, save_path = 'model/', save_prefix= '', drop_extreme_label=True):
        self.n_epochs = n_epochs
        # True for opensource training data; False for original training data (already processed when dumped).
        self.drop_extreme_label = drop_extreme_label
        self.lr = lr
        self.device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.train_stop_loss_thred = train_stop_loss_thred

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
    # Hooks that improved variants may override.
    # ------------------------------------------------------------------
    def _forward(self, feature, inst=None):
        """Run the network. Returns a dict with at least key 'pred' (N,)."""
        return {'pred': self.model(feature)}

    def _train_loss(self, out, label):
        """Returns (loss used for backprop, monitored MSE used by the stop criterion)."""
        loss = self.loss_fn(out['pred'], label)
        return loss, loss

    def _extra_metrics(self, outs, labels):
        """outs/labels: per-day lists (label is cross-sectionally z-scored, NaN kept). Returns dict."""
        return {}

    def _iter_batches(self, dataset, shuffle):
        """Yields (data[N,T,F] tensor, instrument codes[N]) for each trading day."""
        sampler = DailyBatchSamplerRandom(dataset, shuffle)
        instruments = np.asarray(dataset.get_index().get_level_values('instrument'))
        for idx in sampler:
            yield torch.from_numpy(dataset[idx]), instruments[idx]

    def train_epoch(self, dataset):
        self.model.train()
        losses = []

        for data, inst in self._iter_batches(dataset, shuffle=True):
            '''
            data.shape: (N, T, F)
            N - number of stocks
            T - length of lookback_window, 8
            F - 158 factors + 63 market information + 1 label
            '''
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].to(self.device)

            # Additional process on labels
            # If you use original data to train, you won't need the following lines because we already drop extreme when we dumped the data.
            # If you use the opensource data to train, use the following lines to drop extreme labels.
            #########################
            if self.drop_extreme_label:
                mask, label = drop_extreme(label)
                feature = feature[mask, :, :]
                inst = inst[mask.cpu().numpy()]
                label = zscore(label) # CSZscoreNorm
            #########################

            out = self._forward(feature.float(), inst)
            loss, monitor = self._train_loss(out, label)
            losses.append(monitor.item())

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 3.0)
            self.train_optimizer.step()

        return float(np.mean(losses))

    def load_param(self, param_path):
        self.model.load_state_dict(torch.load(param_path, map_location=self.device))
        self.fitted = 'Previously trained.'

    def fit(self, dl_train, dl_valid=None, dl_test=None):
        """dl_test (optional) is only *evaluated* after every epoch for offline analysis of the
        training dynamics; it never influences training or stopping."""
        best_param = None
        self.history = []
        os.makedirs(self.save_path, exist_ok=True)
        for step in range(self.n_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(dl_train)
            self.fitted = step
            rec = {'epoch': step, 'train_loss': train_loss, 'time': time.time() - t0}
            if dl_valid:
                predictions, metrics = self.predict(dl_valid)
                rec.update({f'valid_{k}': float(v) for k, v in metrics.items()})
                print("Epoch %d, train_loss %.6f, valid ic %.4f, icir %.3f, rankic %.4f, rankicir %.3f, %.0fs." % (step, train_loss, metrics['IC'],  metrics['ICIR'],  metrics['RIC'],  metrics['RICIR'], rec['time']), flush=True)
            else: print("Epoch %d, train_loss %.6f" % (step, train_loss), flush=True)
            if dl_test is not None:
                _, tm = self.predict(dl_test)
                rec.update({f'test_{k}': float(v) for k, v in tm.items()})
            self.history.append(rec)

            if train_loss <= self.train_stop_loss_thred:
                best_param = copy.deepcopy(self.model.state_dict())
                torch.save(best_param, f'{self.save_path}/{self.save_prefix}_{self.seed}.pkl')
                break


    def predict(self, dl_test):
        if not isinstance(self.fitted, str) and self.fitted<0:
            raise ValueError("model is not fitted yet!")

        preds = []
        ic = []
        ric = []
        outs, labels = [], []

        self.model.eval()
        for data, inst in self._iter_batches(dl_test, shuffle=False):
            feature = data[:, :, 0:-1].to(self.device)
            label = data[:, -1, -1].numpy()

            # nan label will be automatically ignored when compute metrics.
            # zscorenorm will not affect the results of ranking-based metrics.

            with torch.no_grad():
                out = {k: v.detach().cpu().numpy() for k, v in self._forward(feature.float(), inst).items()}
            pred = out['pred']
            preds.append(pred.ravel())

            daily_ic, daily_ric = calc_ic(pred, label)
            ic.append(daily_ic)
            ric.append(daily_ric)

            # z-score the label cross-sectionally (as during training) for error-based metrics
            valid = ~np.isnan(label)
            zl = np.full_like(label, np.nan)
            if valid.sum() > 1:
                zl[valid] = (label[valid] - label[valid].mean()) / label[valid].std(ddof=1)
            outs.append(out)
            labels.append(zl)

        predictions = pd.Series(np.concatenate(preds), index=dl_test.get_index())

        all_pred = np.concatenate([o['pred'].ravel() for o in outs])
        all_label = np.concatenate(labels)
        v = ~np.isnan(all_label)
        err = all_pred[v] - all_label[v]

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric),
            'MSE': float(np.mean(err ** 2)),
            'MAE': float(np.mean(np.abs(err))),
        }
        metrics.update(self._extra_metrics(outs, labels))

        return predictions, metrics
