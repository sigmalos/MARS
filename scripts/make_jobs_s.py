"""Round-3 job list. Usage: python scripts/make_jobs_s.py <out.jsonl> <phase: inner|fullA|fullB> [fullB config json]"""
import json, os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PY = ['conda', 'run', '--no-capture-output', '-n', 'quant', 'python', '-u', os.path.join(root, 'run_traj.py')]
IC = 'ind,dcorr'
INNER = {  # tag: (relations, model_kwargs, turnover_lambda)
    'base': ('', {}, 0.0),
    'ic_w60': (IC, {'corr_window': 60}, 0.0),
    'ic_w20': (IC, {'corr_window': 20}, 0.0), 'ic_w40': (IC, {'corr_window': 40}, 0.0),
    'ic_w120': (IC, {'corr_window': 120}, 0.0), 'ic_w250': (IC, {'corr_window': 250}, 0.0),
    'e_corrw': (IC, {'corr_edge': 'weight'}, 0.0), 'e_indw': (IC, {'ind_edge': 'weight'}, 0.0),
    'e_inds': (IC, {'ind_edge': 'signed'}, 0.0), 'e_corrs': (IC, {'corr_mode': 'signed', 'corr_edge': 'signed'}, 0.0),
    'e_both': (IC, {'ind_edge': 'signed', 'corr_mode': 'signed', 'corr_edge': 'signed'}, 0.0),
    'to_0.1': (IC, {}, 0.1), 'to_0.3': (IC, {}, 0.3), 'to_1.0': (IC, {}, 1.0),
}
SEEDS_INNER = {'csi300': range(5), 'csi800': range(3)}
FULL_A = {'base': INNER['base'], 'ic_w60': INNER['ic_w60']}


def job(tag, rels, kw, lam, uni, proto, seed):
    jid = f'{tag}_{uni}_{proto}_{seed}'
    cmd = PY + ['--universe', uni, '--seed', str(seed), '--protocol', proto, '--relations', rels, '--model_kwargs', json.dumps(kw),
                '--turnover_lambda', str(lam), '--tag', tag, '--result_dir', os.path.join(root, 'results')]
    return {'id': jid, 'cmd': cmd, 'done': os.path.join(root, 'results', f'{jid}.json')}


phase = sys.argv[2]
jobs = []
if phase == 'inner':
    for uni in ['csi800', 'csi300']:
        for seed in range(5):
            for tag, (r, kw, lam) in INNER.items():
                if seed in SEEDS_INNER[uni]:
                    jobs.append(job(tag, r, kw, lam, uni, 'inner', seed))
else:
    cfgs = FULL_A if phase == 'fullA' else json.load(open(sys.argv[3]))
    for seed in range(10):
        for uni in ['csi800', 'csi300']:
            for tag, (r, kw, lam) in cfgs.items():
                if isinstance(kw, dict) and uni in kw:   # per-universe config (selected on the inner split)
                    r, kw, lam = kw[uni]
                jobs.append(job(tag, r, kw, lam, uni, 'full', seed))
with open(sys.argv[1], 'w') as f:
    for j in jobs:
        f.write(json.dumps(j) + '\n')
print(len(jobs), 'jobs')
