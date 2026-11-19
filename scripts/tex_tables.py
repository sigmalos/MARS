"""Extra LaTeX table fragments (matched epoch, loss matched, uncertainty, sizing, ablation) from analysis_out/final.json."""
import json, os, sys
F = json.load(open(sys.argv[1])); D = sys.argv[2]
N = {'base': 'MASTER', 'ind': '+行业图（CSRC）', 'corr': '+动态相关图', 'fine': '+申万二级图', 'ind_corr': '+行业+相关',
     'multi': '+多关系（三图）', 'rand': '+随机图（对照）'}
def pd_(d, dg=4): return f"{d['diff']:+.{dg}f} [{d['wins']}/{d['n']}] {d['p_t']:.3f}"
# matched epoch
L = []
for t in ['base', 'ind', 'corr', 'fine', 'ind_corr', 'multi', 'rand']:
    c = [N[t]]
    for u in ['csi300', 'csi800']:
        e = F['matched_epoch'][f'{u}/{t}']
        c += [f"{e['IC']:.4f}", f"{e['RIC']:.4f}"] + ([pd_(e['d_IC']), pd_(e['d_RIC'])] if 'd_IC' in e else ['--', '--'])
    L.append(' & '.join(c) + r' \\')
open(f'{D}/matched.tex', 'w').write('\n'.join(L) + '\n')
# loss matched (L = 0.97)
L = []
for t in ['ind', 'corr', 'fine', 'ind_corr', 'multi', 'rand']:
    c = [N[t]]
    for u in ['csi300', 'csi800']:
        e = F['loss_matched'][f'{u}/0.97/{t}']
        c += [f"{e['epoch']:.1f}", pd_(e['d_IC']), pd_(e['d_RIC'])]
    L.append(' & '.join(c) + r' \\')
open(f'{D}/lossmatched.tex', 'w').write('\n'.join(L) + '\n')
# ablation (relation switches), stop protocol + matched epoch, CSI300 / CSI800 dRIC
L = []
for t, sw in [('ind', '✓ & & '), ('corr', ' & ✓ & '), ('fine', ' & & ✓'), ('ind_corr', '✓ & ✓ & '), ('multi', '✓ & ✓ & ✓')]:
    c = [sw.replace('✓', r'\checkmark')]
    for u in ['csi300', 'csi800']:
        s = F[f'{u}/stop'][t]; m = F['matched_epoch'][f'{u}/{t}']
        c += [f"{s['d_IC']['diff']:+.4f}", f"{s['d_RIC']['diff']:+.4f}", f"{m['d_IC']['diff']:+.4f}", f"{m['d_RIC']['diff']:+.4f}"]
    L.append(' & '.join(c) + r' \\')
open(f'{D}/ablation.tex', 'w').write('\n'.join(L) + '\n')
# uncertainty
U = F['uncertainty']
H = [('const_gaussian', '常数 σ', '高斯'), ('hv_gaussian', '历史波动率', '高斯'), ('repr_gaussian', '两阶段（表示）', '高斯'),
     ('const_studentt', '常数 σ', 'Student-t'), ('hv_studentt', '历史波动率', 'Student-t'), ('repr_studentt', '两阶段（表示）', 'Student-t'),
     ('joint_studentt', '联合训练', 'Student-t')]
for u in ['csi300', 'csi800']:
    L = []
    for k, a, b in H:
        e = U[f'{u}/{k}']; g = lambda x, f='.3f': (format(e[x]['mean'], f) if x in e and e[x]['mean'] == e[x]['mean'] else '--')
        L.append(' & '.join([a, b, f"{e['NLL']['mean']:.4f}", g('ECE', '.4f'), g('PICP50'), g('PICP90'), g('PICP99'), g('MPIW90'),
                             g('spearman_sd_abserr'), g('partial_sd_abserr_given_hv'), g('nu', '.2f')]) + r' \\')
        if k in ('repr_gaussian', 'repr_studentt'):
            L.append(r'\midrule')
    open(f'{D}/unc_{u}.tex', 'w').write('\n'.join(L) + '\n')
# sizing
S = F['sizing']; W = [('EW', '等权（官方）'), ('inv_sd', r'$\propto 1/\hat{s}$'), ('inv_sd2', r'$\propto 1/\hat{s}^2$'), ('inv_hv', r'$\propto 1/\mathrm{HV}_{20}$（对照）')]
L = []
for u in ['csi300', 'csi800']:
    for w, name in W:
        a, b = S[f'{u}/nocost/{w}'], S[f'{u}/cost/{w}']
        dv = '--' if w == 'EW' else f"{a['d_excess_vol']['diff']*100:+.2f} [{10 - a['d_excess_vol']['wins']}/10]"
        dir_ = '--' if w == 'EW' else f"{a['d_excess_IR']['diff']:+.2f} ({a['d_excess_IR']['p_t']:.3f})"
        dirc = '--' if w == 'EW' else f"{b['d_excess_IR']['diff']:+.2f} ({b['d_excess_IR']['p_t']:.3f})"
        L.append(' & '.join([u.upper() if w == 'EW' else '', name, f"{a['excess_AR']['mean']:.3f}", f"{a['excess_IR']['mean']:.2f}",
                             f"{a['excess_vol']['mean']*100:.2f}", dv, dir_, f"{b['excess_IR']['mean']:.2f}", dirc, f"{a['turnover']['mean']:.3f}"]) + r' \\')
    if u == 'csi300':
        L.append(r'\midrule')
open(f'{D}/sizing.tex', 'w').write('\n'.join(L) + '\n')
print('ok')
