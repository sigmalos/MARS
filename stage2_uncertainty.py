"""Stage 2 of the decoupled uncertainty model.

The stage-1 network (trained with MSE only) is frozen; its shared representation H (output of the temporal
aggregation, 256-d) and mean prediction mu are extracted once. Only a scale head sigma^2 = Softplus(MLP(H)) + eps
is then trained with the Gaussian or Student-t negative log-likelihood; mu (and therefore the ranking) is unchanged.
Labels are cross-sectionally z-scored over all stocks of a day (no extreme-label dropping, so tails are learned).
Head epoch is selected on the validation NLL. Two reference heads are fitted the same way:
  const : a single learnable scale (and nu)          hv : scale from 20-day historical volatility only

Usage: python stage2_uncertainty.py --universe csi300 --seed 0 --stage1_tag multi --relations ind,corr,fine
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
from torch import nn

from mars.master import MASTERModel
from mars import relations as rel
from mars.uncertainty import ScaleHead, nll_torch, prob_metrics

p = argparse.ArgumentParser()
p.add_argument('--universe', default='csi300')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--stage1_tag', default='multi')
p.add_argument('--relations', default='ind,corr,fine')
p.add_argument('--ckpt_kind', default='stop', choices=['stop', 'best'])
p.add_argument('--ckpt_dir', default='checkpoints')
p.add_argument('--data_dir', default='/data/dataset/MASTER')
p.add_argument('--relation_dir', default='/data/dataset/MASTER_relations')
p.add_argument('--dists', default='gaussian,studentt')
p.add_argument('--epochs', type=int, default=30)
p.add_argument('--patience', type=int, default=5)
p.add_argument('--lr', type=float, default=1e-3)
p.add_argument('--hidden', type=int, default=64)
p.add_argument('--result_dir', default='results_stage2')
p.add_argument('--gpu', type=int, default=0)
args = p.parse_args()
relations = [r for r in args.relations.split(',') if r]
dev = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')


def load(split):
    ds = pickle.load(open(f'{args.data_dir}/opensource/{args.universe}_dl_{split}.pkl', 'rb'))
    rel.attach(ds, args.relation_dir, args.universe, split)
    return ds


m = MASTERModel(d_feat=158, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
                beta=5 if args.universe == 'csi300' else 2, gate_input_start_index=158, gate_input_end_index=221,
                n_epochs=1, lr=1e-5, GPU=args.gpu, seed=args.seed, train_stop_loss_thred=0.95, relations=relations)
m.load_param(f'{args.ckpt_dir}/{args.stage1_tag}/{args.universe}_{args.seed}_{args.ckpt_kind}.pkl')
for q in m.model.parameters():
    q.requires_grad_(False)

feats = {}
for split in ['train', 'valid', 'test']:
    ds = load(split)
    _, out = m.evaluate(ds, return_repr=True)           # eval mode: dropout off, deterministic representation
    db = m.batches(ds)
    day = np.repeat(np.arange(len(db)), db.counts)
    hv = ds.relation_arrays['hv20'].astype(np.float32)
    lhv = np.log(np.where(np.isfinite(hv) & (hv > 0), hv, np.nan))
    for k in range(len(db)):                            # missing hv -> median of the day
        s, n = db.starts[k], db.counts[k]
        seg = lhv[s:s + n]
        seg[~np.isfinite(seg)] = np.nanmedian(seg) if np.isfinite(seg).any() else np.log(0.02)
    feats[split] = {'H': torch.from_numpy(out['repr']), 'mu': torch.from_numpy(out['pred']),
                    'y': torch.from_numpy(out['label_z']), 'lhv': torch.from_numpy(lhv), 'hv': hv,
                    'day': day, 'starts': db.starts, 'counts': db.counts}
    print(split, 'extracted', out['repr'].shape, flush=True)
mu_lhv, sd_lhv = float(feats['train']['lhv'].mean()), float(feats['train']['lhv'].std())


class ConstHead(nn.Module):
    def __init__(self, dist):
        super().__init__()
        self.inner = ScaleHead(1, 1, dist=dist)

    def forward(self, x):
        return self.inner(torch.zeros(x.shape[0], 1, device=x.device))


def head_input(kind, F, sl):
    if kind == 'repr':
        return F['H'][sl]
    return ((F['lhv'][sl] - mu_lhv) / sd_lhv).unsqueeze(-1)


def run_head(kind, dist):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    head = {'repr': lambda: ScaleHead(256, args.hidden, dist=dist), 'hv': lambda: ScaleHead(1, 16, dist=dist),
            'const': lambda: ConstHead(dist)}[kind]().to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    def nll_split(split):
        F = feats[split]
        head.eval()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for s, n in zip(F['starts'], F['counts']):
                sl = slice(s, s + n)
                y = F['y'][sl].to(dev)
                ok = ~torch.isnan(y)
                o = head(head_input(kind, F, sl).to(dev))
                l = nll_torch(y[ok], F['mu'][sl].to(dev)[ok], o['scale'][ok], o['nu'][ok] if 'nu' in o else None)
                tot += l.sum().item()
                cnt += ok.sum().item()
        return tot / cnt

    best, best_state, bad, hist = np.inf, None, 0, []
    for ep in range(args.epochs):
        head.train()
        F = feats['train']
        for k in np.random.permutation(len(F['starts'])):
            s, n = F['starts'][k], F['counts'][k]
            sl = slice(s, s + n)
            y = F['y'][sl].to(dev)
            ok = ~torch.isnan(y)
            o = head(head_input(kind, F, sl).to(dev))
            loss = nll_torch(y[ok], F['mu'][sl].to(dev)[ok], o['scale'][ok], o['nu'][ok] if 'nu' in o else None).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = nll_split('valid')
        hist.append(v)
        if v < best - 1e-5:
            best, best_state, bad = v, {k_: t.clone() for k_, t in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    F = feats['test']
    scale = np.zeros(len(F['mu']), np.float32)
    nu = np.zeros(len(F['mu']), np.float32) if dist == 'studentt' else None
    with torch.no_grad():
        for s, n in zip(F['starts'], F['counts']):
            sl = slice(s, s + n)
            o = head(head_input(kind, F, sl).to(dev))
            scale[sl] = o['scale'].cpu().numpy()
            if nu is not None:
                nu[sl] = o['nu'].cpu().numpy()
    met = prob_metrics(F['mu'].numpy(), scale, F['y'].numpy(), nu=nu, hv=F['hv'], day=F['day'])
    met.update({'valid_nll_history': hist, 'best_valid_nll': best, 'epochs_run': len(hist)})
    return met, scale, nu


os.makedirs(args.result_dir, exist_ok=True)
res = {'universe': args.universe, 'seed': args.seed, 'stage1_tag': args.stage1_tag, 'ckpt_kind': args.ckpt_kind}
arrays = {}
for dist in args.dists.split(','):
    for kind in ['repr', 'hv', 'const']:
        met, scale, nu = run_head(kind, dist)
        res[f'{kind}_{dist}'] = met
        print(f'{kind:5s} {dist:8s} NLL {met["NLL"]:.4f} PICP90 {met["PICP90"]:.3f} ECE {met["ECE"]:.4f} '
              f'rho(sd,|e|) {met["spearman_sd_abserr"]:.3f} partial|hv {met["partial_sd_abserr_given_hv"]:.3f}', flush=True)
        if kind == 'repr':
            arrays[f'scale_{dist}'] = scale
            if nu is not None:
                arrays[f'nu_{dist}'] = nu
arrays['mu'] = feats['test']['mu'].numpy()
arrays['label_z'] = feats['test']['y'].numpy()
arrays['hv20'] = feats['test']['hv']
np.savez_compressed(f'{args.result_dir}/{args.stage1_tag}_{args.universe}_{args.seed}_{args.ckpt_kind}_stage2.npz', **arrays)
json.dump(res, open(f'{args.result_dir}/{args.stage1_tag}_{args.universe}_{args.seed}_{args.ckpt_kind}_stage2.json', 'w'),
          indent=1, default=float)
