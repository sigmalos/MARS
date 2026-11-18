from master import MASTERModel
import argparse
import json
import os
import pickle
import numpy as np
import time

# Please install qlib first before load the data.

parser = argparse.ArgumentParser()
parser.add_argument('--universe', default='csi300', choices=['csi300', 'csi800'])
parser.add_argument('--prefix', default='opensource', choices=['original', 'opensource'],
                    help='which training data to use (valid/test always come from opensource)')
parser.add_argument('--data_dir', default='/data/dataset/MASTER', help='directory containing original/ and opensource/')
parser.add_argument('--mode', default='eval', choices=['train', 'eval'],
                    help='train: train from scratch; eval: load checkpoints from --model_dir and test')
parser.add_argument('--seeds', default='0', help='comma separated, e.g. 0,1,2,3,4')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_epoch', type=int, default=40)
parser.add_argument('--model_dir', default='model')
parser.add_argument('--save_dir', default='model', help='where trained checkpoints are written')
parser.add_argument('--result_file', default=None, help='optional json file to dump per-seed metrics')
parser.add_argument('--stop_loss', type=float, default=0.95, help='training-loss stop threshold (0 = run all n_epoch)')
parser.add_argument('--test_each_epoch', action='store_true', help='record test metrics after every epoch (analysis only)')
parser.add_argument('--model_kwargs', default='{}', help='extra JSON kwargs forwarded to MASTERModel (variant switches)')
args = parser.parse_args()

universe = args.universe
prefix = args.prefix
seeds = [int(s) for s in args.seeds.split(',')]

if args.mode == 'train':
    with open(os.path.join(args.data_dir, prefix, f'{universe}_dl_train.pkl'), 'rb') as f:
        dl_train = pickle.load(f)

predict_data_dir = os.path.join(args.data_dir, 'opensource')
with open(os.path.join(predict_data_dir, f'{universe}_dl_valid.pkl'), 'rb') as f:
    dl_valid = pickle.load(f)
with open(os.path.join(predict_data_dir, f'{universe}_dl_test.pkl'), 'rb') as f:
    dl_test = pickle.load(f)

print("Data Loaded.", flush=True)


d_feat = 158
d_model = 256
t_nhead = 4
s_nhead = 2
dropout = 0.5
gate_input_start_index = 158
gate_input_end_index = 221

if universe == 'csi300':
    beta = 5
elif universe == 'csi800':
    beta = 2

n_epoch = args.n_epoch
lr = 1e-5
GPU = args.gpu
train_stop_loss_thred = args.stop_loss
# Original training data already went through DropExtreme + CSZScoreNorm on labels when dumped.
drop_extreme_label = (prefix == 'opensource')


ic = []
icir = []
ric = []
ricir = []
records = []

for seed in seeds:
    model = MASTERModel(
        d_feat = d_feat, d_model = d_model, t_nhead = t_nhead, s_nhead = s_nhead, T_dropout_rate=dropout, S_dropout_rate=dropout,
        beta=beta, gate_input_end_index=gate_input_end_index, gate_input_start_index=gate_input_start_index,
        n_epochs=n_epoch, lr = lr, GPU = GPU, seed = seed, train_stop_loss_thred = train_stop_loss_thred,
        save_path=args.save_dir, save_prefix=f'{universe}_{prefix}', drop_extreme_label=drop_extreme_label,
        **json.loads(args.model_kwargs)
    )

    start = time.time()
    if args.mode == 'train':
        model.fit(dl_train, dl_valid, dl_test if args.test_each_epoch else None)
        print("Model Trained.", flush=True)
    else:
        param_path = os.path.join(args.model_dir, f'{universe}_{prefix}_{seed}.pkl')
        print(f'Model Loaded from {param_path}')
        model.load_param(param_path)

    predictions, metrics = model.predict(dl_test)
    running_time = time.time() - start

    print('Seed: {:d} time cost : {:.2f} sec'.format(seed, running_time))
    print(metrics, flush=True)

    ic.append(metrics['IC'])
    icir.append(metrics['ICIR'])
    ric.append(metrics['RIC'])
    ricir.append(metrics['RICIR'])
    records.append({'seed': seed, 'epoch': model.fitted, 'time': running_time,
                    'n_params': sum(p.numel() for p in model.model.parameters()),
                    **{k: float(v) for k, v in metrics.items()},
                    'history': getattr(model, 'history', [])})

print("IC: {:.4f} pm {:.4f}".format(np.mean(ic), np.std(ic)))
print("ICIR: {:.4f} pm {:.4f}".format(np.mean(icir), np.std(icir)))
print("RIC: {:.4f} pm {:.4f}".format(np.mean(ric), np.std(ric)))
print("RICIR: {:.4f} pm {:.4f}".format(np.mean(ricir), np.std(ricir)))

if args.result_file:
    os.makedirs(os.path.dirname(args.result_file) or '.', exist_ok=True)
    with open(args.result_file, 'w') as f:
        json.dump({'universe': universe, 'prefix': prefix, 'mode': args.mode,
                   'model_kwargs': json.loads(args.model_kwargs), 'runs': records}, f, indent=2)
