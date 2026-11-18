"""Figures for docs/main.tex (static PDF, print). Palette: validated reference categorical slots 1-3 on white
(blue #2a78d6, orange #eb6834, aqua #1baf7a; aqua < 3:1 contrast -> every chart carries direct labels / a table
in the paper). Baseline / reference marks use neutral gray ink. One y-axis per panel, thin marks, recessive grid.

Usage: python scripts/figures.py --summary analysis_out/summary.json --results results --stage2 results_stage2 --out ../../MASTER/docs/figures
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
NAMES = {'base': 'MASTER', 'ind': '+Ind', 'corr': '+Corr', 'fine': '+SW2', 'ind_corr': '+Ind+Corr', 'multi': '+Multi',
         'rand': '+Rand', 'snap': '+Snap', 'sw1': '+SW1', 'sw3': '+SW3', 'multi_jt': '+Multi+JointT'}

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK SC', 'DejaVu Sans'], 'font.size': 8,
    'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'axes.titlesize': 8.5,
    'axes.titlecolor': INK, 'axes.unicode_minus': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
})


def ci95(d):
    from scipy import stats
    d = np.asarray(d)
    return stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


def fig_paired(summary, runs, out, metric='RIC', proto='stop', tags=('ind', 'corr', 'fine', 'ind_corr', 'multi', 'rand')):
    """Dot + 95% CI of the paired difference vs MASTER, one panel per universe (same metric, shared zero line)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=True, gridspec_kw={'wspace': 0.42})
    for ax, uni in zip(axes, ['csi300', 'csi800']):
        ys = np.arange(len(tags))[::-1]
        for y, tag in zip(ys, tags):
            a, b = runs.get((uni, tag)), runs.get((uni, 'base'))
            if not a or not b:
                continue
            common = sorted(set(a) & set(b))
            d = np.array([a[s][proto]['test'][metric] - b[s][proto]['test'][metric] for s in common])
            col = MUTED if tag == 'rand' else BLUE
            ax.errorbar(d.mean(), y, xerr=ci95(d), fmt='o', ms=4.5, color=col, ecolor=col, elinewidth=1.2, capsize=0)
            ax.scatter(d, np.full(len(d), y) + 0.22, s=4, color=col, alpha=0.45, linewidths=0)
            ax.text(1.02, y, f'{d.mean():+.4f} [{(d > 0).sum()}/{len(d)}]', transform=ax.get_yaxis_transform(), ha='left',
                    va='center', fontsize=6.5, color=INK2, clip_on=False)
        ax.axvline(0, color=AXIS, lw=0.8, zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels([NAMES[t] for t in tags])
        ax.set_title(uni.upper())
        ax.set_xlabel(f'Δ{"RankIC" if metric == "RIC" else metric} vs MASTER（同种子配对）')
        ax.grid(axis='y', visible=False)
    fig.savefig(out)
    plt.close(fig)


def fig_curves(runs, out, uni='csi300', tags=('base', 'ind', 'multi')):
    """Mean test RankIC per epoch (all seeds); one panel, one axis. Thin 1.5pt lines, direct end labels."""
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    cols = {'base': MUTED, 'ind': BLUE, 'multi': ORANGE, 'corr': AQUA}
    ends = []
    for tag in tags:
        seeds = runs.get((uni, tag), {})
        if not seeds:
            continue
        L = min(len(r['history']) for t in tags for r in runs.get((uni, t), {}).values())
        M = np.array([[h['test_RIC'] for h in r['history'][:L]] for r in seeds.values()])
        x = np.arange(1, L + 1)
        ax.plot(x, M.mean(0), color=cols[tag], lw=1.5)
        ends.append([M.mean(0)[-1], NAMES[tag], x[-1]])
    ends.sort()
    gap = 0.0024 if uni == 'csi300' else 0.0018
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for yv, name, xl in ends:
        ax.text(xl + 0.4, yv, name, color=INK2, fontsize=6.8, va='center')
    ax.set_xlabel('训练轮数（epoch）')
    ax.set_ylabel('测试集 RankIC（种子均值）')
    ax.set_title(f'{uni.upper()} 训练动态')
    fig.savefig(out)
    plt.close(fig)


def fig_calibration(final, out):
    """Calibration error (empirical - nominal coverage of central intervals), one panel per universe, one axis each.
    Plotting the deviation instead of the raw reliability curve makes the few-percent differences visible."""
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4), sharey=True)
    series = [('const_gaussian', MUTED, '常数σ 高斯', '--'), ('repr_gaussian', BLUE, '两阶段 高斯', '-'),
              ('repr_studentt', ORANGE, '两阶段 Student-t', '-'), ('joint_studentt', AQUA, '联合训练 Student-t', '-')]
    for ax, uni in zip(axes, ['csi300', 'csi800']):
        ax.axhline(0, color=AXIS, lw=0.8)
        for key, col, lab, ls in series:
            cal = final['uncertainty'][f'{uni}/{key}']['calibration']
            lv = sorted(float(k) for k in cal)
            dev = [cal[str(l)] - l if str(l) in cal else cal[l] - l for l in lv]
            ax.plot(lv, dev, color=col, lw=1.5, ls=ls, marker='o', ms=3.2, label=lab)
        ax.set_title(uni.upper())
        ax.set_xlabel('名义覆盖率')
    axes[0].set_ylabel('实际 − 名义覆盖率')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, fontsize=6.8, bbox_to_anchor=(0.5, -0.15))
    fig.savefig(out)
    plt.close(fig)


def fig_matched(final, out, metric='RIC', tags=('ind', 'corr', 'fine', 'ind_corr', 'multi', 'rand')):
    """Paired difference vs MASTER of the test metric averaged over the matched epoch window (all models trained
    the same number of epochs). Dot = mean, bar = 95% CI, small dots = seeds."""
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=True, gridspec_kw={'wspace': 0.42})
    for ax, uni in zip(axes, ['csi300', 'csi800']):
        base = np.array(final['matched_epoch'][f'{uni}/base']['lists'][metric])
        ys = np.arange(len(tags))[::-1]
        for y, tag in zip(ys, tags):
            d = np.array(final['matched_epoch'][f'{uni}/{tag}']['lists'][metric]) - base
            col = MUTED if tag == 'rand' else BLUE
            ax.errorbar(d.mean(), y, xerr=ci95(d), fmt='o', ms=4.5, color=col, ecolor=col, elinewidth=1.2, capsize=0)
            ax.scatter(d, np.full(len(d), y) + 0.22, s=4, color=col, alpha=0.45, linewidths=0)
            ax.text(1.02, y, f'{d.mean():+.4f} [{(d > 0).sum()}/{len(d)}]', transform=ax.get_yaxis_transform(), ha='left',
                    va='center', fontsize=6.5, color=INK2, clip_on=False)
        r = final['matched_epoch'][f'{uni}/base']['range']
        ax.axvline(0, color=AXIS, lw=0.8, zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels([NAMES[t] for t in tags])
        ax.set_title(f'{uni.upper()}（第 {r[0]}–{r[1]} 轮平均）')
        ax.set_xlabel(f'Δ{"RankIC" if metric == "RIC" else metric} vs MASTER（同种子、同训练轮数）')
        ax.grid(axis='y', visible=False)
    fig.savefig(out)
    plt.close(fig)


def fig_rel_alpha(results, out):
    """Average relation-attention weight of the multi-relation model per universe (grouped bars, labelled)."""
    fig, ax = plt.subplots(figsize=(3.0, 2.3))
    rel_names = ['行业(CSRC)', '相关性', '申万二级']
    cols = [BLUE, ORANGE, AQUA]
    for j, uni in enumerate(['csi300', 'csi800']):
        files = sorted(glob.glob(os.path.join(results, f'multi_{uni}_*_stop_pred.npz')))
        if not files:
            continue
        A = np.mean([np.load(f)['rel_alpha'].mean(0) for f in files], axis=0)
        for r in range(3):
            x = j * 4 + r
            ax.bar(x, A[r], width=0.86, color=cols[r], edgecolor='white', linewidth=1.0)
            ax.text(x, max(A[r], (1 / 3 + 0.008) * (abs(A[r] - 1 / 3) < 0.02)) + 0.008, f'{A[r]:.3f}', ha='center', fontsize=6.5, color=INK2)
    ax.set_xticks([1, 5])
    ax.set_xticklabels(['CSI300', 'CSI800'])
    ax.set_ylabel('平均关系注意力权重 α')
    ax.axhline(1 / 3, color=AXIS, lw=0.7, ls=(0, (3, 3)), zorder=0)
    ax.text(7.55, 1 / 3, '均匀 1/3', fontsize=6, color=MUTED, va='bottom', ha='right')
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in cols], rel_names, fontsize=6.5, loc='upper center',
              ncol=3, bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis='x', visible=False)
    fig.savefig(out)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results', default='results')
    p.add_argument('--stage2', default='results_stage2')
    p.add_argument('--out', default='figures')
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = {}
    for f in glob.glob(os.path.join(a.results, '*.json')):
        r = json.load(open(f))
        if 'stop' in r:
            runs.setdefault((r['universe'], r['tag']), {})[r['seed']] = r
    fig_paired(None, runs, os.path.join(a.out, 'paired_ric.pdf'), 'RIC')
    fig_paired(None, runs, os.path.join(a.out, 'paired_ic.pdf'), 'IC')
    fig_paired(None, runs, os.path.join(a.out, 'paired_ric_best.pdf'), 'RIC', proto='best')
    fig_curves(runs, os.path.join(a.out, 'curves_csi300.pdf'), 'csi300')
    fig_curves(runs, os.path.join(a.out, 'curves_csi800.pdf'), 'csi800')
    fig_rel_alpha(a.results, os.path.join(a.out, 'rel_alpha.pdf'))
    final = json.load(open(os.path.join(os.path.dirname(a.results.rstrip('/')) or '.', 'analysis_out', 'final.json')))
    fig_calibration(final, os.path.join(a.out, 'calibration.pdf'))
    fig_matched(final, os.path.join(a.out, 'matched_ric.pdf'), 'RIC')
    fig_matched(final, os.path.join(a.out, 'matched_ic.pdf'), 'IC')
    print('figures written to', a.out)
