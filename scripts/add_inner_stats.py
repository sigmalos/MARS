"""Add the inner-split rule parameters (E*, L*, K*) of configurations trained after the selection (phase-B inner
runs, e.g. final_g) to analysis_out/selection.json. All earlier choices (rule family, W*, edge, lambda*, n_drop*)
are left unchanged. Usage: python scripts/add_inner_stats.py final_g
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import traj as T  # noqa: E402

R, OUT = 'results', 'analysis_out/selection.json'
sel = json.load(open(OUT))
runs = T.load_runs(R, 'inner')
for tag in sys.argv[1:]:
    for uni in ['csi300', 'csi800']:
        seeds = runs.get((uni, tag))
        if not seeds:
            continue
        inst, dt, y = T.index(R, uni, 'ival')
        groups = T.day_groups(dt)
        S = sorted(seeds)
        ic = np.array([[h['ival_IC'] for h in seeds[s]['history']] for s in S])
        ric = np.array([[h['ival_RIC'] for h in seeds[s]['history']] for s in S])
        loss = np.array([[h['train_loss'] for h in seeds[s]['history']] for s in S])
        sm = T.smooth(ic.mean(0))
        E = int(np.argmax(sm))
        e_off = [T.first_below(l, T.OFFICIAL) for l in loss]
        L_val = {L: float(np.mean([ic[i, T.first_below(loss[i], L)] for i in range(len(S))])) for L in T.LOSS_GRID}
        L_star = max(L_val, key=L_val.get)
        P = {s: T.preds(R, seeds[s], 'ival') for s in S}
        ens = {}
        for K in T.ENS_K:
            vals = [T.daily_ic(T.ensemble(P[s], groups, E, K), y, groups) for s in S]
            ens[K] = {m: float(np.mean([v[m] for v in vals])) for m in ['IC', 'RIC', 'ICIR', 'RICIR']}
        K_star = max(ens, key=lambda k: ens[k]['IC'])
        sel[uni]['stats'][tag] = {
            'seeds': S, 'n_params': seeds[S[0]]['n_params'], 'E_star': E, 'L_star': L_star, 'K_star': K_star,
            'off': {'epoch': float(np.mean(e_off)), 'IC': float(np.mean([ic[i, e] for i, e in enumerate(e_off)])),
                    'RIC': float(np.mean([ric[i, e] for i, e in enumerate(e_off)]))},
            'epoch': {'epoch': E, 'IC': float(ic[:, E].mean()), 'IC_smooth': float(sm[E]), 'RIC': float(ric[:, E].mean())},
            'loss': {'L': L_star, 'epoch': float(np.mean([T.first_below(l, L_star) for l in loss])), 'IC': L_val[L_star],
                     'RIC': float(np.mean([ric[i, T.first_below(loss[i], L_star)] for i in range(len(S))]))},
            'ens': {'K': K_star, **ens[K_star]}, 'ens_all': ens,
            'curve_IC': ic.mean(0).tolist(), 'curve_RIC': ric.mean(0).tolist(), 'curve_loss': loss.mean(0).tolist(),
            'added_after_selection': True}
        st = sel[uni]['stats'][tag]
        print(f"{uni} {tag}: off IC {st['off']['IC']:.4f} | E*={E} {st['epoch']['IC']:.4f} | L*={L_star} {st['loss']['IC']:.4f} | "
              f"K*={K_star} {st['ens']['IC']:.4f} RIC {st['ens']['RIC']:.4f}")
json.dump(sel, open(OUT, 'w'), indent=1, default=float)
