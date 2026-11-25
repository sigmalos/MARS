# AGENTS.md

Guidance for coding agents working in this repository (MARS: Market-Aware Relational Stock Transformer, a
relation-enhanced fork of MASTER, AAAI 2024). Read README.md for the model, results and data preparation.

## Layout

- `mars/` — core library, imported as a package (`from mars.master import MASTERModel`, `from mars import relations`).
  - `master.py` model (MASTER backbone + multi-relation GAT, relation attention, gated fusion, on-the-fly correlation graphs)
  - `base_model.py` training / evaluation loop; `relations.py` attaches relation arrays; `uncertainty.py`; `backtest.py`
- Entry points at the repo root: `main.py` (train/eval, records both stopping protocols), `run_traj.py` (fixed-length
  trajectories, inner split), `stage2_uncertainty.py`. Run them from the repo root so that `mars` is importable.
- `scripts/` — data building, job generation / queue, selection, statistics, backtests, figures. Scripts that import
  `mars` put the repo root on `sys.path` themselves. `scripts/traj.py` is a helper module, not a script.
- `results_summary/` — aggregated results (JSON) that back the reported numbers; keep in sync when re-running analyses.
- `qlib-update/` — reference files from the upstream MASTER repo; not used by the code.

## Data and models (not in git)

- `data/` -> `/data/dataset/MARS` and `model/` -> `/data/models/MARS` are **symlinks on this machine** and are git-ignored
  (`/data`, `/model` in `.gitignore`). Never commit them or replace the symlinks with real files.
- Other inputs (absolute default paths in the scripts, override with `--data_dir`, `--relation_dir`, `--qlib_dir`):
  - `/data/dataset/MASTER/opensource/{csi300,csi800}_dl_{train,valid,test}.pkl` — official MASTER data (qlib TSDataSampler)
  - `/data/dataset/MASTER_relations/` — per-sample relation arrays, `returns.npz`, raw industry downloads
  - `/data/dataset/qlib_cn_chenditc/extracted/qlib_bin` — Qlib daily data (prices, benchmarks)
- `docs/`, `logs/`, `results/`, `checkpoints/`, `analysis_out/`, `results_stage2/`, `.claude/` are git-ignored outputs.

## Environment

- Use the conda env `quant`: `conda run --no-capture-output -n quant python ...` (Python 3.11, torch 2.5, pyqlib 0.9.7).
  Do not install packages into it without asking; isolate extra packages (e.g. `pip install --target`).
- 8 × RTX 3090. When launching many runs set `OMP_NUM_THREADS` (2–8): the default thread count oversubscribes the
  CPU and slows every run ~2×. Use `scripts/job_queue.py --global_cap 40`; it counts all training processes on the
  machine. Never name a module `queue.py` (shadows the stdlib and breaks pandas imports).
- GitHub is only reachable through the local proxy: run `clashctl on` (shell function from `~/.bashrc`) in the same
  shell command as `git push` / `gh`, and do not pipe it (the proxy variables would be set in a subshell).

## Experimental protocol (keep it)

- Every comparison is **paired by seed**: the same seed gives identical initialisation of the shared parameters and
  the same day order. New modules must be created after the original layers, be zero-initialised (or copy an
  existing matrix) and must not consume the global RNG, so that a new switch set to its neutral value reproduces the
  previous model exactly. Check this with a forward-equivalence test (max |diff| = 0) before launching runs.
- Report mean ± std over seeds (10 for final results), paired t-test / Wilcoxon, Holm correction over the main
  family, and the number of improving seeds.
- **The test set is never used for selection.** Hyper-parameters and stopping rules are chosen on the inner split
  (train ≤ 2017-12-22, validate 2018-01-02 … 2020-03-31) or fixed in advance; the official validation set (67 days)
  is too short for selection. Per-epoch test metrics are for analysing training dynamics only.
- The official stopping rule (train MSE ≤ 0.95) stops larger models earlier; always also report a matched-epoch
  comparison when model capacity differs.
- Relation data must be point-in-time: only information effective on or before the sample date; correlation windows
  use returns strictly before day t.

## Checks before reporting results

- `python -m py_compile mars/*.py *.py scripts/*.py`
- Official checkpoint evaluation must reproduce IC 0.0651 / RankIC 0.0685 on CSI300:
  `python main.py --mode eval --universe csi300 --seeds 0 --tag official --ckpt "model/csi300_opensource_{seed}.pkl" --result_dir /tmp/eval`
- Backtests use Qlib `backtest_daily` (TopkDropout top-30, close price, default costs buy 0.05% / sell 0.15%).
