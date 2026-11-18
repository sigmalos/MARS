"""Generate the job list of the formal experiments (seed-major order so every configuration progresses evenly)."""
import json, os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PY = ['conda', 'run', '--no-capture-output', '-n', 'quant', 'python', '-u', os.path.join(root, 'main.py')]
BOTH = {  # tag: (relations, model_kwargs)
    'base': ('', {}), 'ind': ('ind', {}), 'corr': ('corr', {}), 'fine': ('fine', {}),
    'ind_corr': ('ind,corr', {}), 'multi': ('ind,corr,fine', {}), 'rand': ('rand', {}),
    'multi_jt': ('ind,corr,fine', {'joint_unc': 'studentt'}),
}
CSI300_ONLY = {'snap': ('snap', {}), 'sw1': ('sw1', {}), 'sw3': ('sw3', {})}
seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
jobs = []
for seed in seeds:
    for uni in ['csi800', 'csi300']:
        cfgs = dict(BOTH, **(CSI300_ONLY if uni == 'csi300' else {}))
        for tag, (rels, kw) in cfgs.items():
            jid = f'{tag}_{uni}_{seed}'
            cmd = PY + ['--universe', uni, '--seeds', str(seed), '--tag', tag, '--relations', rels, '--model_kwargs', json.dumps(kw),
                        '--result_dir', os.path.join(root, 'results'), '--ckpt_dir', os.path.join(root, 'checkpoints')]
            jobs.append({'id': jid, 'cmd': cmd, 'done': os.path.join(root, 'results', f'{jid}.json')})
with open(sys.argv[1], 'w') as f:
    for j in jobs:
        f.write(json.dumps(j) + '\n')
print(len(jobs), 'jobs')
