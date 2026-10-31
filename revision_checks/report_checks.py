"""Build tables and query-budget figures from stored results only."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/muon_revision_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS = ROOT/'results'
METHODS = ['sampled_coordinate', 'jaguar', 'subspace_muon', 'rank_matched_sgd']
LABELS = dict(sampled_coordinate='Sampled-coordinate Muon', jaguar='JAGUAR Muon (SVD)',
              subspace_muon='Subspace ZO-Muon (SVD)', rank_matched_sgd='Norm-matched ZO-SGD')


def regression_figures():
    methods = ['Entrywise DFO-Muon', 'Low-rank DFO-Muon', 'Low-rank DFO-Muon v2',
               'Direct structured search']
    labels = ['Entrywise', 'Random rank-one', 'Basis-aligned V2', 'Direct search']
    colors = ['#0072B2', '#009E73', '#D55E00', '#CC79A7']
    styles = ['-', '--', ':', '-.']
    for case in range(1, 5):
        with (PROJECT/'muon_zero_order_reproduction/results'/f'matrix_regression_case_{case}.csv').open() as f:
            records = list(csv.DictReader(f))
        fig, ax = plt.subplots(figsize=(3.0, 2.24), layout='constrained')
        for method, label, color, style in zip(methods, labels, colors, styles):
            rows = [r for r in records if r['method'] == method]
            ax.loglog([int(r['function_evaluations']) for r in rows],
                      [float(r['relative_objective']) for r in rows],
                      label=label, color=color, ls=style, lw=1.3)
        ax.set(xlabel='Cumulative function evaluations', ylabel='Normalized objective gap',
               title=f'Case {case}', ylim=(1e-4, 1.4))
        ax.grid(alpha=.2, which='both')
        ax.legend(frameon=False, fontsize=7.1, loc='lower left')
        fig.savefig(RESULTS/f'matrix_regression_query_case_{case}.pdf')
        fig.savefig(RESULTS/f'matrix_regression_query_case_{case}.png', dpi=160)
        plt.close(fig)


def collect():
    records = [json.loads(p.read_text()) for p in RESULTS.glob('*.json')
               if p.name.startswith(('test_', 'tune_', 'timing_', 'sensitivity_'))
               and not p.name.endswith('_environment.json')]
    summary = dict(methods={}, sensitivity=[], tuning_calls=0, evaluation_calls=0,
                   sensitivity_calls=0, serial_timing_calls=0)
    for phase, key in [('tune', 'tuning_calls'), ('test', 'evaluation_calls'),
                       ('sensitivity', 'sensitivity_calls'), ('timing', 'serial_timing_calls')]:
        selected = [r for r in records if r['phase'] == phase]
        summary[key] = sum(r['calls'] for r in selected)
        summary[phase+'_runs'] = len(selected)
    for views in [20, 120]:
        summary['methods'][str(views)] = {}
        for method in METHODS:
            rows = [r for r in records if r['phase'] == 'test'
                    and r['views'] == views and r['method'] == method]
            assert len(rows) == 5, (views, method, len(rows))
            gaps = np.array([r['gap'] for r in rows])
            record = dict(eta=rows[0]['eta'], calls=rows[0]['calls'],
                          iterations=rows[0]['iterations'], median_gap=float(np.median(gaps)),
                          gap_quartiles=np.quantile(gaps, [.25, .75]).tolist(),
                          median_residual=float(np.median([r['data_residual'] for r in rows])))
            if views == 20:
                record['median_heldout_residual'] = float(np.median([r['heldout_residual'] for r in rows]))
            timing = [r for r in records if r['phase'] == 'timing'
                      and r['views'] == views and r['method'] == method]
            if timing:
                record['serial_times'] = timing[0]['times']
            summary['methods'][str(views)][method] = record
    conditions = sorted({(r['q'], r['radius']) for r in records if r['phase'] == 'sensitivity'})
    for q, radius in conditions:
        rows = [r for r in records if r['phase'] == 'sensitivity'
                and r['q'] == q and r['radius'] == radius]
        assert len(rows) == 5
        summary['sensitivity'].append(dict(q=q, radius=radius, eta=.01,
                                           median_gap=float(np.median([r['gap'] for r in rows])),
                                           gap_quartiles=np.quantile([r['gap'] for r in rows], [.25, .75]).tolist(),
                                           calls=rows[0]['calls'], iterations=rows[0]['iterations']))
    (RESULTS/'additional_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary, records


def additional_table(summary):
    lines = [r'\begin{table}[tbp]', r'\centering\color{revisionred}', r'\footnotesize',
             r'\caption{Additional walnut CT controls under the same 20,000-call ceiling. Entries are median normalized gaps [Q1, Q3] over five evaluation seeds. Calls include initialization and endpoint values. SVD denotes the stated adaptation of the cited method.}',
             r'\label{tab:additional-controls}', r'\begin{tabular}{lrrr}', r'\toprule',
             r'Method & Calls & 20 views & 120 views \\', r'\midrule']
    for method in METHODS:
        parts = []
        for views in [20, 120]:
            r = summary['methods'][str(views)][method]
            parts.append(f"{r['median_gap']:.3f} [{r['gap_quartiles'][0]:.3f}, {r['gap_quartiles'][1]:.3f}]")
        lines.append(f"{LABELS[method]} & {r['calls']:,} & {parts[0]} & {parts[1]} "+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (RESULTS/'additional_table.tex').write_text('\n'.join(lines)+'\n')


def additional_figure(records):
    colors = ['#0072B2', '#D55E00', '#CC79A7', '#009E73']
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), layout='constrained')
    for views, ax in zip([20, 120], axes):
        reference = json.loads((PROJECT/'realdata_ct/results'/f'reference_{views}.json').read_text())['objective']
        for method, color in zip(METHODS, colors):
            rows = [r for r in records if r['phase'] == 'test'
                    and r['views'] == views and r['method'] == method]
            histories = []
            for r in sorted(rows, key=lambda r: r['seed']):
                stem = f"test_{views}_{method}_{r['seed']}_{r['eta']:g}_q32_h0.001".replace('.', 'p')
                histories.append(np.load(RESULTS/(stem+'.npz'))['history'])
            gaps = np.array([(h[:, 2]-reference)/(.5-reference) for h in histories])
            ax.semilogy(histories[0][:, 0], np.median(gaps, axis=0), label=LABELS[method],
                        color=color, lw=1.3)
            low, high = np.quantile(gaps, [.25, .75], axis=0)
            ax.fill_between(histories[0][:, 0], low, high, color=color, alpha=.14, lw=0)
        ax.set(xlabel='Cumulative function evaluations', ylabel='Normalized objective gap',
               title=f'{views} measured views')
        ax.grid(alpha=.2, which='both')
        if views == 20:
            ax.legend(frameon=False, fontsize=6.4)
    fig.savefig(RESULTS/'additional_budget_controls.pdf')
    fig.savefig(RESULTS/'additional_budget_controls.png', dpi=180)
    plt.close(fig)


def main():
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    RESULTS.mkdir(exist_ok=True)
    regression_figures()
    summary, records = collect()
    additional_table(summary)
    additional_figure(records)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
