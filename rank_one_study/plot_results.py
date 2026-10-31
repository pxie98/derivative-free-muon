"""Plot fresh regression runs and the saved CartPole episode records."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent/'results'


def main():
    with (OUT/'trajectories.csv').open() as handle:
        records = list(csv.DictReader(handle))
    plt.rcParams.update({'font.size': 10.5, 'pdf.fonttype': 42})
    for case in range(1, 5):
        rows = [r for r in records if int(r['case']) == case]
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        coord = [r for r in rows if r['method'] == 'Coordinate']
        ax.loglog([int(r['calls']) for r in coord],
                  [float(r['relative_gap']) for r in coord],
                  color='.35', linestyle='--', label='Coordinate recovery')
        seeds = sorted({int(r['seed']) for r in rows if r['method'] != 'Coordinate'})
        paths = [[r for r in rows if int(r['seed']) == seed] for seed in seeds]
        gaps = np.array([[float(r['relative_gap']) for r in path] for path in paths])
        calls = [int(r['calls']) for r in paths[0]]
        ax.loglog(calls, np.median(gaps, axis=0), color='blue', linewidth=1.8,
                  label='RO-DFO-Muon (30 seeds)')
        lo, hi = np.quantile(gaps, [.1, .9], axis=0)
        ax.fill_between(calls, lo, hi, color='blue', alpha=.14, linewidth=0)
        ax.axhline(.1, color='.6', linestyle=':', linewidth=.8)
        ax.set(xlabel='Function evaluations', ylabel='Relative objective gap', title=f'Case {case}')
        ax.grid(True, which='both', alpha=.2)
        ax.legend(fontsize=8.5, loc='lower left')
        fig.tight_layout()
        fig.savefig(OUT/f'query_case_{case}.pdf')
        plt.close(fig)

    with (ROOT/'muon_zero_order_reproduction/results/cartpole_dfo/trajectory.csv').open() as handle:
        records = list(csv.DictReader(handle))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    styles = [('Policy-gradient Muon', '.3', '--'), ('Rank-one ZO-SGD', 'tab:orange', ':'),
              ('DFO-Muon', 'blue', '-'), ('Enhanced DFO-Muon', 'tab:green', '-.')]
    for method, color, linestyle in styles:
        rows = [r for r in records if r['method'] == method]
        seeds = sorted({int(r['seed']) for r in rows})
        paths = [sorted([r for r in rows if int(r['seed']) == seed],
                        key=lambda r: int(r['iteration'])) for seed in seeds]
        episodes = np.array([[int(r['episodes']) for r in path] for path in paths])
        assert np.all(episodes == episodes[0])
        returns = np.array([[float(r['return']) for r in path] for path in paths])
        mean, std = returns.mean(axis=0), returns.std(axis=0)
        ax.plot(episodes[0], mean, label=('RO-DFO-Muon (layerwise)' if method == 'DFO-Muon' else method), color=color, linestyle=linestyle, linewidth=1.6)
        ax.fill_between(episodes[0], mean-std, mean+std, color=color, alpha=.12, linewidth=0)
    ax.axhline(475, color='.5', linestyle=':', linewidth=.8)
    ax.set(xlabel='Training episodes', ylabel='Mean evaluation return', ylim=(0, 520))
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'cartpole_episodes.pdf')
    plt.close(fig)


if __name__ == '__main__':
    main()
