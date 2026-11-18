"""Minimal resumable job queue for many single-GPU training runs.

jobs file: JSON lines {"id", "cmd" (list), "done" (file that exists when finished)}; jobs whose `done` file exists
are skipped. Each job gets CUDA_VISIBLE_DEVICES of the GPU with the fewest running jobs.
Usage: python scripts/queue.py jobs.jsonl --max_jobs 40 --gpus 0,1,2,3,4,5,6,7 --threads 2 --log_dir logs
"""
import argparse
import json
import os
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('jobs')
p.add_argument('--max_jobs', type=int, default=40)
p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
p.add_argument('--threads', type=int, default=2)
p.add_argument('--log_dir', default='logs')
args = p.parse_args()

gpus = [int(g) for g in args.gpus.split(',')]
jobs = [json.loads(l) for l in open(args.jobs) if l.strip()]
todo = [j for j in jobs if not os.path.exists(j['done'])]
print(f'{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to run', flush=True)
os.makedirs(args.log_dir, exist_ok=True)


def gpu_load():
    """Number of compute processes per GPU (all users/queues), so several queues can share the machine."""
    try:
        idx = dict(l.split(', ') for l in subprocess.check_output(
            ['nvidia-smi', '--query-gpu=uuid,index', '--format=csv,noheader'], text=True).strip().splitlines())
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid', '--format=csv,noheader'], text=True).split()
        load = {g: 0 for g in gpus}
        for u in apps:
            if int(idx[u]) in load:
                load[int(idx[u])] += 1
        return load
    except Exception:
        return {g: 0 for g in gpus}


running = {}  # id -> (Popen, gpu, t0)
failed = []
while todo or running:
    for jid, (proc, g, t0) in list(running.items()):
        if proc.poll() is not None:
            ok = proc.returncode == 0
            print(f'{time.strftime("%H:%M:%S")} {"done" if ok else "FAILED"} {jid} rc={proc.returncode} {time.time() - t0:.0f}s', flush=True)
            if not ok:
                failed.append(jid)
            del running[jid]
    while todo and len(running) < args.max_jobs:
        job = todo.pop(0)
        load = gpu_load()
        for _, g, t0 in running.values():   # just-started jobs may not hold a CUDA context yet
            if time.time() - t0 < 90:
                load[g] += 1
        g = min(gpus, key=lambda x: load[x])
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
        log = open(os.path.join(args.log_dir, f'{job["id"]}.log'), 'w')
        running[job['id']] = (subprocess.Popen(job['cmd'], stdout=log, stderr=subprocess.STDOUT, env=env), g, time.time())
        print(f'{time.strftime("%H:%M:%S")} start {job["id"]} on GPU {g} ({len(running)} running, {len(todo)} queued)', flush=True)
    time.sleep(10)
print('ALL DONE; failed:', failed, flush=True)
