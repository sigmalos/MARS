"""Train / evaluate (relation-enhanced) MASTER.

Example:
  python main.py --universe csi300 --seeds 0 --relations ind,corr,fine --tag multi --result_dir results
Every training run lasts --n_epoch epochs and records two model-selection protocols from the same trajectory:
  stop : official criterion (first epoch with train MSE <= 0.95)      best : best validation IC
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

from mars.master import MASTERModel
from mars import relations as rel

parser = argparse.ArgumentParser()
parser.add_argument('--universe', default='csi300', choices=['csi300', 'csi800'])
parser.add_argument('--prefix', default='opensource', choices=['original', 'opensource'], help='training data source')
parser.add_argument('--data_dir', default='/data/dataset/MASTER')
parser.add_argument('--relation_dir', default='/data/dataset/MASTER_relations')
parser.add_argument('--mode', default='train', choices=['train', 'eval'])
parser.add_argument('--seeds', default='0', help='comma separated, e.g. 0,1,2')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_epoch', type=int, default=40)
parser.add_argument('--stop_loss', type=float, default=0.95)
parser.add_argument('--official_stop', action='store_true', help='end training at the official criterion')
parser.add_argument('--select_metric', default='IC')
parser.add_argument('--patience', type=int, default=8, help='early stopping for the best-validation protocol (<0: off)')
parser.add_argument('--relations', default='', help='comma separated subset of ' + ','.join(rel_name for rel_name in
                    ['ind', 'corr', 'fine', 'sw1', 'sw3', 'snap', 'rand']))
parser.add_argument('--model_kwargs', default='{}', help='extra JSON kwargs for MASTERModel (e.g. joint_unc)')
parser.add_argument('--tag', default='base')
parser.add_argument('--ckpt', default=None, help='eval mode: checkpoint path pattern with {seed}')
parser.add_argument('--result_dir', default='results')
parser.add_argument('--ckpt_dir', default='checkpoints')
args = parser.parse_args()

universe, prefix = args.universe, args.prefix
seeds = [int(s) for s in args.seeds.split(',')]
relations = [r for r in args.relations.split(',') if r]
model_kwargs = json.loads(args.model_kwargs)


def load(split, source):
    with open(os.path.join(args.data_dir, source, f'{universe}_dl_{split}.pkl'), 'rb') as f:
        ds = pickle.load(f)
    if relations and source == 'opensource':
        rel.attach(ds, args.relation_dir, universe, split)
    return ds


dl_train = load('train', prefix) if args.mode == 'train' else None
dl_valid = load('valid', 'opensource')
dl_test = load('test', 'opensource')
print("Data Loaded.", flush=True)

d_feat = 158
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 158
gate_input_end_index = 221
beta = 5 if universe == 'csi300' else 2
lr = 1e-5

os.makedirs(args.result_dir, exist_ok=True)
for seed in seeds:
    model = MASTERModel(
        d_feat=d_feat, d_model=d_model, t_nhead=t_nhead, s_nhead=s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=args.n_epoch, lr=lr, GPU=args.gpu, seed=seed, train_stop_loss_thred=args.stop_loss,
        save_path=os.path.join(args.ckpt_dir, args.tag), save_prefix=universe, drop_extreme_label=(prefix == 'opensource'),
        select_metric=args.select_metric, relations=relations, **model_kwargs)
    n_params = sum(p.numel() for p in model.model.parameters())
    start = time.time()
    record = {'tag': args.tag, 'universe': universe, 'prefix': prefix, 'seed': seed, 'relations': relations,
              'model_kwargs': model_kwargs, 'n_params': n_params}
    keep = ('pred', 'scale', 'nu', 'rel_alpha')
    if args.mode == 'train':
        selected = model.fit(dl_train, dl_valid, dl_test, full_epochs=not args.official_stop, keep_outputs=keep,
                             patience=args.patience if args.patience >= 0 else None)
        record['history'] = model.history
        for name, sel in selected.items():
            record[name] = {'epoch': sel['epoch'], 'test': sel['test'], 'valid': sel['valid']}
            np.savez_compressed(os.path.join(args.result_dir, f'{args.tag}_{universe}_{seed}_{name}_pred.npz'),
                                **{k: v.astype(np.float32) for k, v in sel['outputs'].items()})
    else:
        model.load_param(args.ckpt.format(seed=seed))
        vm, _ = model.evaluate(dl_valid)
        tm, outputs = model.evaluate(dl_test)
        record['eval'] = {'test': tm, 'valid': vm}
        np.savez_compressed(os.path.join(args.result_dir, f'{args.tag}_{universe}_{seed}_eval_pred.npz'),
                            **{k: v.astype(np.float32) for k, v in outputs.items() if k in keep or k == 'label_z'})
    record['time'] = time.time() - start
    summary = record.get('stop', record.get('eval', {})).get('test', {})
    print('Seed %d (%s) %.0fs: %s' % (seed, args.tag, record['time'], {k: round(v, 4) for k, v in summary.items()}), flush=True)
    with open(os.path.join(args.result_dir, f'{args.tag}_{universe}_{seed}.json'), 'w') as f:
        json.dump(record, f, indent=1, default=float)

# test-set index (shared by all prediction files of this universe)
idx_file = os.path.join(args.result_dir, f'index_{universe}_test.npz')
if not os.path.exists(idx_file):
    index = dl_test.get_index()
    np.savez_compressed(idx_file, instrument=np.asarray(index.get_level_values('instrument')).astype('U12'),
                        datetime=index.get_level_values('datetime').values.astype('datetime64[D]'))
