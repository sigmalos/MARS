"""Fixed-length training run that records the whole trajectory (round-3 experiments).

protocol 'inner' : train on training-set days <= 2017-12-22, evaluate on 2018-01-02 .. 2020-03-31 ('ival') each
                   epoch (hyper-parameter / stopping-rule selection; no test-set access).
protocol 'full'  : train on the full training set, evaluate the official valid ('valid') and test ('test') sets.
Outputs: <result_dir>/<tag>_<uni>_<proto>_<seed>.json (config, history) and ..._preds.npz (float16 [epochs, n]
predictions per evaluation set). Shared indices / z-scored labels: index_<uni>_<set>.npz.
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

from master import MASTERModel
import relations as rel

p = argparse.ArgumentParser()
p.add_argument('--universe', default='csi300', choices=['csi300', 'csi800'])
p.add_argument('--seed', type=int, default=0)
p.add_argument('--protocol', default='full', choices=['inner', 'full'])
p.add_argument('--epochs', type=int, default=40)
p.add_argument('--relations', default='')
p.add_argument('--model_kwargs', default='{}')
p.add_argument('--turnover_lambda', type=float, default=0.0)
p.add_argument('--tag', default='base')
p.add_argument('--gpu', type=int, default=0)
p.add_argument('--data_dir', default='/data/dataset/MASTER')
p.add_argument('--relation_dir', default='/data/dataset/MASTER_relations')
p.add_argument('--result_dir', default='results')
args = p.parse_args()
INNER_TRAIN, INNER_VAL = (None, '2017-12-22'), ('2018-01-02', None)   # 5 trading days gap: labels use close_{t+5}

relations = [r for r in args.relations.split(',') if r]
kw = json.loads(args.model_kwargs)
need_ret = 'dcorr' in relations or kw.get('ind_edge', 'none') != 'none'


def load(split):
    ds = pickle.load(open(f'{args.data_dir}/opensource/{args.universe}_dl_{split}.pkl', 'rb'))
    if relations:
        rel.attach(ds, args.relation_dir, args.universe, split, returns=need_ret)
    return ds


dl_train = load('train')
if args.protocol == 'inner':
    train, evals = (dl_train, INNER_TRAIN), {'ival': (dl_train, INNER_VAL)}
else:
    train, evals = (dl_train, None), {'valid': (load('valid'), None), 'test': (load('test'), None)}
print('Data Loaded.', flush=True)

model = MASTERModel(
    d_feat=158, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
    beta=5 if args.universe == 'csi300' else 2, gate_input_start_index=158, gate_input_end_index=221,
    n_epochs=args.epochs, lr=1e-5, GPU=args.gpu, seed=args.seed, train_stop_loss_thred=0.95,
    save_path='checkpoints', save_prefix=args.universe, drop_extreme_label=True,
    turnover_lambda=args.turnover_lambda, relations=relations, **kw)
t0 = time.time()
history, preds = model.fit_trajectory(train, evals, args.epochs)
os.makedirs(args.result_dir, exist_ok=True)
name = f'{args.tag}_{args.universe}_{args.protocol}_{args.seed}'
np.savez_compressed(os.path.join(args.result_dir, f'{name}_preds.npz'), **preds)
for set_name, (ds, rng) in evals.items():
    f = os.path.join(args.result_dir, f'index_{args.universe}_{set_name}.npz')
    if not os.path.exists(f):
        ix = model.batches(ds, rng).index()
        tmp = f'{f[:-4]}.{os.getpid()}.tmp.npz'   # atomic: several runs may finish at the same time
        np.savez_compressed(tmp, instrument=np.asarray(ix.get_level_values('instrument')).astype('U12'),
                            datetime=ix.get_level_values('datetime').values.astype('datetime64[D]'),
                            label_z=model.eval_labels[set_name])
        os.replace(tmp, f)
rec = {'tag': args.tag, 'universe': args.universe, 'protocol': args.protocol, 'seed': args.seed, 'relations': relations,
       'model_kwargs': kw, 'turnover_lambda': args.turnover_lambda, 'epochs': args.epochs,
       'n_params': sum(q.numel() for q in model.model.parameters()), 'train_days': len(model.batches(*train)),
       'history': history, 'time': time.time() - t0}
json.dump(rec, open(os.path.join(args.result_dir, f'{name}.json'), 'w'), indent=1, default=float)
print('done', name, '%.0fs' % rec['time'], flush=True)
