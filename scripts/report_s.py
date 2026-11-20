"""LaTeX table fragments and figures for the round-3 chapter of docs/main.tex.

Inputs: analysis_out/selection.json (inner split) and analysis_out/final_s.json (test set).
Usage: python scripts/report_s.py --tex_dir ../../MASTER/docs/tables3 --fig_dir ../../MASTER/docs/figures3
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--analysis', default='analysis_out')
p.add_argument('--tex_dir', default='tables3')
p.add_argument('--fig_dir', default='figures3')
a = p.parse_args()
os.makedirs(a.tex_dir, exist_ok=True)
os.makedirs(a.fig_dir, exist_ok=True)
SEL = json.load(open(os.path.join(a.analysis, 'selection.json')))
F = json.load(open(os.path.join(a.analysis, 'final_s.json')))
U = ['csi300', 'csi800']
RULE_NAME = {'off': '官方阈值 0.95', 'vbest': '官方验证集最优', 'epoch': '固定轮数 $E^*$', 'loss': '损失阈值 $L^*$',
             'ens': '轮次集成 $(E^*,K^*)$'}
TAG = {'base': 'MASTER', 'ic_w60': '+行业+相关（$W$=60）', 'ic_wstar': '+行业+相关（$W$=20）', 'final_g': '+行业+相关（$W$=20，带符号边）'}
EDGE = {'ic_w60': '无权重（基准）', 'e_corrw': '相关图：相关系数加权', 'e_indw': '行业图：残差相关加权', 'e_inds': '行业图：带符号',
        'e_corrs': '相关图：带符号（$k^+$=10，$k^-$=5）', 'e_both': '两图均带符号'}
W = {'ic_w20': 20, 'ic_w40': 40, 'ic_w60': 60, 'ic_w120': 120, 'ic_w250': 250}
LAM = {'ic_w60': 0.0, 'to_0.1': 0.1, 'to_0.3': 0.3, 'to_1.0': 1.0}


def pf(d, dg=4):
    return f"{d['diff']:+.{dg}f} [{d['wins']}/{d['n']}] {d['p_t']:.3f}" + (r'$^{*}$' if d.get('p_holm', 1) < 0.05 else '')


# -------------------------------------------------- inner split: window / edge / lambda (selected rule's IC)
def inner_table(tags, names, fname):
    lines = []
    for t in tags:
        c = [names[t]]
        for uni in U:
            st = SEL[uni]['stats'].get(t)
            r = SEL[uni]['rule']
            c += [f"{st['off']['IC']:.4f}", f"{st[r]['IC']:.4f}", f"{st[r].get('RIC', st['ens']['RIC']):.4f}"] if st else ['--'] * 3
        lines.append(' & '.join(map(str, c)) + r' \\')
    open(os.path.join(a.tex_dir, fname), 'w').write('\n'.join(lines) + '\n')


inner_table(list(W), {t: str(w) for t, w in W.items()}, 'inner_window.tex')
inner_table(list(EDGE), EDGE, 'inner_edge.tex')
inner_table(['base'] + list(LAM), {'base': 'MASTER（无图）', **{t: f'{l:g}' for t, l in LAM.items()}}, 'inner_lambda.tex')

# stopping rules on the inner split (mean over all inner configurations) + selected parameters per model
lines = []
for rule in ['off', 'epoch', 'loss', 'ens']:
    c = [RULE_NAME[rule]] + [f"{SEL[u]['rule_mean_IC'][rule]:.4f}" for u in U]
    lines.append(' & '.join(c) + r' \\')
open(os.path.join(a.tex_dir, 'inner_rules.tex'), 'w').write('\n'.join(lines) + '\n')
lines = []
for t in ['base', 'ic_w60', 'ic_w20', 'final_g']:
    c = [TAG.get(t, TAG['ic_wstar'] if t == 'ic_w20' else t)]
    for uni in U:
        st = SEL[uni]['stats'][t]
        c += [f"{st['off']['epoch']:.1f}", str(st['E_star']), f"{st['L_star']:.3f}", str(st['K_star'])]
    lines.append(' & '.join(c) + r' \\')
open(os.path.join(a.tex_dir, 'inner_params.tex'), 'w').write('\n'.join(lines) + '\n')

# inner n_drop / lambda backtests
lines = []
for uni in U:
    ib = SEL[uni]['inner_backtest']
    for nd in sorted(ib['n_drop'], key=lambda x: -int(x)):
        v = ib['n_drop'][nd]
        lines.append(' & '.join([uni.upper() if nd == max(ib['n_drop'], key=int) else '', str(nd), f"{v['turnover']:.3f}", f"{v['nocost_IR']:.2f}",
                                 f"{v['cost_IR']:.2f}" + (r'$^{\dagger}$' if int(nd) == SEL[uni]['n_drop_star'] else '')]) + r' \\')
    if uni == 'csi300':
        lines.append(r'\midrule')
open(os.path.join(a.tex_dir, 'inner_ndrop.tex'), 'w').write('\n'.join(lines) + '\n')

# -------------------------------------------------- test set
S = F['summary']
# (1) stopping rules on the test set
lines = []
for rule in ['off', 'vbest', 'epoch', 'loss', 'ens']:
    c = [RULE_NAME[rule]]
    for uni in U:
        for tag in ['base', 'ic_w60']:
            v = S[uni]['S'][f'{tag}/{rule}']
            c.append(f"{v['IC']['mean']:.4f}")
        t = S[uni]['tests']
        c.append('--' if rule == 'off' else pf(t[f'rule/ic_w60/{rule}-off/IC']))
        c.append(pf(t[f'graph/{rule}/ic_w60-base/IC']))
    lines.append(' & '.join(c) + r' \\')
open(os.path.join(a.tex_dir, 'test_rules.tex'), 'w').write('\n'.join(lines) + '\n')
# (2) main test table: every configuration under the selected rule
for uni in U:
    r = S[uni]['rule']
    lines = []
    for tag in S[uni]['tags']:
        v = S[uni]['S'][f'{tag}/{r}']
        c = [TAG[tag]] + [f"{v[m]['mean']:.4f}$\\pm${v[m]['std']:.4f}" if m in ('IC', 'RIC') else f"{v[m]['mean']:.3f}" for m in ['IC', 'ICIR', 'RIC', 'RICIR']]
        t = S[uni]['tests']
        c += ['--', '--'] if tag == 'base' else [pf(t[f'vsbase/{tag}/IC']), pf(t[f'vsbase/{tag}/RIC'])]
        lines.append(' & '.join(c) + r' \\')
    ob = S[uni]['S']['base/off']
    lines.insert(0, ' & '.join(['MASTER（官方阈值）'] + [f"{ob[m]['mean']:.4f}$\\pm${ob[m]['std']:.4f}" if m in ('IC', 'RIC') else f"{ob[m]['mean']:.3f}"
                                                        for m in ['IC', 'ICIR', 'RIC', 'RICIR']] + ['--', '--']) + r' \\')
    lines.insert(1, r'\midrule')
    open(os.path.join(a.tex_dir, f'test_main_{uni}.tex'), 'w').write('\n'.join(lines) + '\n')
# (3) step-wise contribution
lines = []
for step, name in [('ic_wstar-ic_w60', '窗口 60 → 20'), ('final_g-ic_wstar', '+ 带符号边')]:
    c = [name]
    for uni in U:
        t = S[uni]['tests']
        c += [pf(t[f'step/{step}/IC']), pf(t[f'step/{step}/RIC'])] if f'step/{step}/IC' in t else ['--', '--']
    lines.append(' & '.join(c) + r' \\')
open(os.path.join(a.tex_dir, 'test_steps.tex'), 'w').write('\n'.join(lines) + '\n')
# (4) backtests
lines = []
for uni in U:
    r, nds = S[uni]['rule'], S[uni]['n_drop_star']
    first = True
    for tag in S[uni]['tags']:
        for rule, nd in [('off', 30), (r, 30), (r, nds)]:
            k = f'{tag}/{rule}/bt{nd}'
            if k not in S[uni]['S']:
                continue
            v = S[uni]['S'][k]
            lines.append(' & '.join([uni.upper() if first else '', TAG[tag] if (rule, nd) == ('off', 30) else '', RULE_NAME[rule].split(' ')[0],
                                     str(nd), f"{v['turnover']['mean']:.3f}", f"{v['nocost_AR']['mean']:.3f}", f"{v['nocost_IR']['mean']:.2f}",
                                     f"{v['cost_AR']['mean']:.3f}", f"{v['cost_IR']['mean']:.2f}$\\pm${v['cost_IR']['std']:.2f}",
                                     f"{v['cost_MDD']['mean']:.3f}"]) + r' \\')
            first = False
    if uni == 'csi300':
        lines.append(r'\midrule')
open(os.path.join(a.tex_dir, 'test_backtest.tex'), 'w').write('\n'.join(lines) + '\n')

# -------------------------------------------------- figures (validated categorical slots 1-3 + neutral gray)
BLUE, ORANGE, AQUA, INK2, MUTED, GRID, AXIS = '#2a78d6', '#eb6834', '#1baf7a', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Noto Sans CJK SC', 'DejaVu Sans'], 'font.size': 8,
                     'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False, 'axes.titlesize': 8.5,
                     'axes.unicode_minus': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight'})
# window: inner IC vs W (selected rule), one panel per universe
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3))
for ax, uni in zip(axes, U):
    r = SEL[uni]['rule']
    xs = list(W.values())
    for key, col, lab in [('off', MUTED, '官方阈值'), (r, BLUE, '选定规则')]:
        ys = [SEL[uni]['stats'][t][key]['IC'] for t in W]
        ax.plot(xs, ys, color=col, lw=1.5, marker='o', ms=3.5, label=lab)
        ax.text(xs[-1] * 1.08, ys[-1], lab, color=INK2, fontsize=6.8, va='center')
    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlim(16, 420)
    ax.set_xlabel('相关窗口 W（交易日）')
    ax.set_title(f'{uni.upper()}（内部验证集）')
axes[0].set_ylabel('IC')
fig.savefig(os.path.join(a.fig_dir, 'window.pdf'))
plt.close(fig)
# test curves of the four configurations (mean over seeds) with the official stop marked
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.4))
cols = {'base': MUTED, 'ic_w60': BLUE, 'ic_wstar': AQUA, 'final_g': ORANGE}
short = {'base': 'MASTER', 'ic_w60': 'W=60', 'ic_wstar': 'W=20', 'final_g': 'W=20+符号'}
for ax, uni in zip(axes, U):
    ends = []
    for tag in S[uni]['tags']:
        C = np.array([F['metrics'][k]['IC'] for k in F['metrics'] if k.startswith(f'{uni}/{tag}/curve/')])
        m = np.convolve(C.mean(0), np.ones(3) / 3, mode='valid')
        x = np.arange(2, len(m) + 2)
        ax.plot(x, m, color=cols[tag], lw=1.4)
        ends.append([m[-1], short[tag], x[-1]])
        off = np.mean([F['epochs'][k]['off'] for k in F['epochs'] if k.startswith(f'{uni}/{tag}/')])
        ax.plot([off + 1], [np.interp(off + 1, x, m)], marker='v', color=cols[tag], ms=5)
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + (0.0016 if uni == 'csi300' else 0.0013))
    for yv, name, xl in ends:
        ax.text(xl + 0.6, yv, name, color=INK2, fontsize=6.6, va='center')
    ax.set_xlim(0, 50)
    ax.set_xlabel('训练轮数（epoch）')
    ax.set_title(f'{uni.upper()}（测试集）')
axes[0].set_ylabel('测试集 IC（10 种子均值）')
fig.savefig(os.path.join(a.fig_dir, 'test_curves.pdf'))
plt.close(fig)
# n_drop trade-off on the inner split: cost IR vs turnover
fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3))
for ax, uni in zip(axes, U):
    nd = SEL[uni]['inner_backtest']['n_drop']
    ks = sorted(nd, key=int)
    ax.plot([nd[k]['turnover'] for k in ks], [nd[k]['cost_IR'] for k in ks], color=BLUE, lw=1.5, marker='o', ms=3.5)
    for k in ks:
        ax.annotate(f'n_drop={k}', (nd[k]['turnover'], nd[k]['cost_IR']), textcoords='offset points', xytext=(5, -9 if k == '3' and uni == 'csi800' else 4), fontsize=6.5, color=INK2)
    ax.set_xlabel('日均换手率（Qlib，买卖双边）')
    ax.margins(x=0.15, y=0.12)
    ax.set_title(f'{uni.upper()}（内部验证集，含费）')
axes[0].set_ylabel('含费信息比率 IR')
fig.savefig(os.path.join(a.fig_dir, 'ndrop.pdf'))
plt.close(fig)
print('ok')
