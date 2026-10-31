"""Regenerate manuscript artifacts from saved CT runs, without fitting data."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/muon_ct_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_walnut import ROOT, METHODS, TEST_SEEDS, problem

RESULTS = ROOT/'results'
LABELS = dict(rankone='Rank-one Muon', orthogonal='Orthogonal Muon',
              sphere='Spherical Muon', zo_sgd='Normalized ZO-SGD',
              coordinate='Coordinate Muon')
COLORS = dict(rankone='#0072B2', orthogonal='#009E73', sphere='#CC79A7',
              zo_sgd='#D55E00', coordinate='#555555')


def name(phase, v, m, s, e):
    return f'{phase}_{v}_{m}_{s}_{e:g}'.replace('.', 'p')


def plot_comparison(summary, histories, methods, image_methods, images, sinogram,
                    attenuation_limits, filename):
    """Plot saved iterates with a shared scale and explicitly selected controls."""
    plot_labels = dict(LABELS, rankone='RO-DFO-Muon',
                       coordinate='Full coordinate recovery')
    fig = plt.figure(figsize=(7.1, 4.7), layout='constrained')
    grid = fig.add_gridspec(2, 4, height_ratios=[1.2, 1.0])
    for col, views in enumerate([20, 120]):
        ax = fig.add_subplot(grid[0, col*2:col*2+2])
        reference = summary[str(views)]['reference']['objective']
        for method in methods:
            runs = histories[views, method]
            gaps = np.array([(h[:, 2]-reference)/(.5-reference) for h in runs])
            calls = runs[0][:, 0]
            median = np.median(gaps, axis=0)
            ax.plot(calls, median, label=plot_labels[method], color=COLORS[method],
                    lw=1.3, marker='o' if method == 'coordinate' else None,
                    markersize=3, drawstyle='steps-post' if method == 'coordinate' else 'default')
            if len(runs) > 1:
                low, high = np.quantile(gaps, [.25, .75], axis=0)
                ax.fill_between(calls, low, high, color=COLORS[method],
                                alpha=.17, linewidth=0)
        ax.set(yscale='log', xlabel='Forward evaluations',
               ylabel='Normalized objective gap', title=f'{views} measured views',
               xlim=(0, 20500), ylim=(.01, 1.1))
        ax.grid(alpha=.2, which='both')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        if views == 20:
            ax.legend(loc='lower left', frameon=False)
    ax = fig.add_subplot(grid[1, 0])
    ax.imshow(sinogram, cmap='gray', aspect='auto', interpolation='nearest')
    ax.set(title='Measured sinogram', xlabel='20 retained views', ylabel='Detector bin')
    image_axes = []
    for index, method in enumerate(['reference', *image_methods], 1):
        ax = fig.add_subplot(grid[1, index])
        image_axes.append(ax)
        im = ax.imshow(images[method], cmap='gray', vmin=attenuation_limits[0],
                       vmax=attenuation_limits[1], interpolation='nearest')
        ax.set_title('CG reference' if method == 'reference' else plot_labels[method])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=image_axes, shrink=.7, pad=.02, label='Attenuation (data units)')
    fig.savefig(RESULTS/(filename+'.pdf'), bbox_inches='tight')
    fig.savefig(RESULTS/(filename+'.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    selected=json.loads((RESULTS/'selected_steps.json').read_text())
    summary={}
    histories={}
    for v in [20,120]:
        ref=json.loads((RESULTS/f'reference_{v}.json').read_text())
        summary[str(v)]={'reference':ref, 'methods':{}}
        for m in METHODS:
            step=selected[str(v)][m]
            stems=[name('test',v,m,s,step) for s in (TEST_SEEDS if m!='coordinate' else [901])]
            runs=[json.loads((RESULTS/(p+'.json')).read_text()) for p in stems]
            hs=[np.load(RESULTS/(p+'.npz'))['history'] for p in stems]
            histories[v,m]=hs
            gaps=np.array([(r['objective']-ref['objective'])/(.5-ref['objective']) for r in runs])
            record=dict(eta=step,calls=runs[0]['calls'],iterations=runs[0]['iterations'],
                        gap_median=float(np.median(gaps)), gap_quartiles=np.quantile(gaps,[.25,.75]).tolist(),
                        residual_median=float(np.median([r['data_residual'] for r in runs])))
            if v==20:
                record['heldout_median']=float(np.median([r['heldout_residual'] for r in runs]))
            timing_path=RESULTS/(name('timing',v,m,901,step)+'.json')
            if timing_path.exists():
                timing=json.loads(timing_path.read_text())
                record['serial_seconds']=timing['seconds']
                record['serial_polar_seconds']=timing['polar_seconds']
            summary[str(v)]['methods'][m]=record
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,
                         'legend.fontsize':6.7,'pdf.fonttype':42,'ps.fonttype':42})
    n, _, observations, _, _, _ = problem(20)
    images = {'reference': np.load(RESULTS/'reference_20.npz')['w']}
    for method in ['rankone', 'coordinate']:
        path = RESULTS/(name('test', 20, method, 901, selected['20'][method])+'.npz')
        images[method] = np.load(path)['w']
    # All three shown reconstructions share the same attenuation range.
    attenuation_limits = (min(float(w.min()) for w in images.values()),
                          max(float(w.max()) for w in images.values()))
    sinogram = observations.reshape(n, 20, order='F')
    plot_comparison(summary, histories, ['rankone', 'coordinate'],
                    ['rankone', 'coordinate'], images, sinogram,
                    attenuation_limits, 'walnut_results')

    lines=[r'\begin{table}[tbp]',r'\centering\color{revisionblue}',
           r'\caption{Measured walnut CT results. Gaps and fitted-data residuals are medians over five evaluation seeds; brackets give gap quartiles. Coordinate recovery is deterministic. Times are separate serial reruns with seed 901; all randomized methods use 19,956 forward calls and coordinate recovery uses 13,450.}',
           r'\label{tab:walnut}',r'\footnotesize',r'\begin{tabular}{clrrrr}',r'\toprule',
           r'Views & Method & $\eta_0$ & $g_T$ [Q1, Q3] & $r_T$ & Time (s) \\',r'\midrule']
    for v in [20,120]:
        if v==120: lines.append(r'\midrule')
        for m in METHODS:
            r=summary[str(v)]['methods'][m]
            gap=f"{r['gap_median']:.3f}"
            if m!='coordinate':gap+=f" [{r['gap_quartiles'][0]:.3f}, {r['gap_quartiles'][1]:.3f}]"
            sec=f"{r['serial_seconds']:.1f}" if 'serial_seconds' in r else r'---'
            lines.append(f"{v} & {LABELS[m]} & {r['eta']:.3g} & {gap} & {r['residual_median']:.3f} & {sec} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',
              r'\begin{figure}[tbp]',r'\centering\color{revisionblue}',
              r'\includegraphics[width=\textwidth]{realdata_ct/results/walnut_results.pdf}',
              r'\caption{Comparison of RO-DFO-Muon with full coordinate recovery on measured walnut CT. Top: normalized objective gaps under a 20,000-call ceiling; the RO-DFO-Muon curve and band show the median and quartiles over five seeds, and coordinate recovery is deterministic; markers denote completed updates. Bottom: the 20-view sinogram, CG reference, RO-DFO-Muon, and coordinate reconstructions for the preassigned seed 901. Reconstruction images share one attenuation scale without positivity clipping. The complete five-method results, including normalized ZO-SGD, are reported in Table~\ref{tab:walnut}.}',
              r'\label{fig:walnut}',r'\end{figure}', '']
    (RESULTS/'results_table_figure.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
