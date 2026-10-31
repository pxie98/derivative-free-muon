"""Recompute table statistics and check every saved query-count sequence."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent/'results'


def main():
    cfg = json.loads((ROOT/'muon_zero_order_reproduction/config.json').read_text())
    summary = json.loads((OUT/'summary.json').read_text())
    with (OUT/'trajectories.csv').open() as handle:
        records = list(csv.DictReader(handle))
    report = dict(status='PASS', random_runs=0, coordinate_runs=0, rows=len(records),
                  targets=summary['targets'], target_point_one_successes=0)
    for case, saved in zip(cfg['cases'], summary['cases']):
        assert saved['case'] == case['id']
        paths = {}
        for r in records:
            if int(r['case']) == case['id']:
                paths.setdefault((r['method'], int(r['seed'])), []).append(r)
        assert len(paths) == 31
        hits = {target: [] for target in summary['targets']}
        coordinate = {}
        terminal = []
        for (method, seed), rows in paths.items():
            rows.sort(key=lambda r: int(r['iteration']))
            assert len(rows) == 201
            assert [int(r['iteration']) for r in rows] == list(range(201))
            calls = np.array([int(r['calls']) for r in rows])
            gaps = np.array([float(r['relative_gap']) for r in rows])
            cost = case['m']*case['n']+1 if method == 'Coordinate' else case['low_rank_probes']+1
            np.testing.assert_array_equal(calls, 1+cost*np.arange(201))
            assert np.isfinite(gaps).all() and gaps[0] == 1
            if method == 'Coordinate':
                report['coordinate_runs'] += 1
            else:
                assert method == 'RO-DFO-Muon' and seed in summary['seeds']
                report['random_runs'] += 1
                terminal.append(gaps[-1])
            for target in summary['targets']:
                indices = np.flatnonzero(gaps <= target)
                hit = int(calls[indices[0]]) if len(indices) else None
                if method == 'Coordinate':
                    coordinate[target] = hit
                elif hit is not None:
                    hits[target].append(hit)
        assert np.median(terminal) == saved['terminal_gap_median']
        for row in saved['targets']:
            target = row['target']
            reached = hits[target]
            assert row['successes'] == len(reached)
            assert row['coordinate_calls'] == coordinate[target]
            median = float(np.median(reached)) if reached else None
            assert row['median_calls_successful'] == median
            assert row['ratio'] == (coordinate[target]/median if median else None)
            assert row['min_calls'] == (min(reached) if reached else None)
            assert row['max_calls'] == (max(reached) if reached else None)
            if target == .1:
                report['target_point_one_successes'] += len(reached)
    assert report['random_runs'] == report['target_point_one_successes'] == 120
    assert report['coordinate_runs'] == 4
    # Figure 4 must preserve the recorded episode budgets and table values.
    with (ROOT/'muon_zero_order_reproduction/results/cartpole_dfo/trajectory.csv').open() as handle:
        cartpole = list(csv.DictReader(handle))
    expected = {'Policy-gradient Muon': 979.2, 'Rank-one ZO-SGD': 2707.2,
                'DFO-Muon': 1641.6, 'Enhanced DFO-Muon': 1305.6}
    for method, episodes_to_solve in expected.items():
        rows = [r for r in cartpole if r['method'] == method]
        seeds = sorted({int(r['seed']) for r in rows})
        assert len(seeds) == 5
        solved = []
        for seed in seeds:
            path = sorted([r for r in rows if int(r['seed']) == seed],
                          key=lambda r: int(r['iteration']))
            assert len(path) == 71
            assert all(int(r['episodes']) == 48*int(r['iteration']) for r in path)
            assert float(path[-1]['return']) == 500
            solved.append(next(int(r['episodes']) for r in path if float(r['return']) >= 475))
        assert np.isclose(np.mean(solved), episodes_to_solve)
    report['cartpole_saved_rows'] = len(cartpole)
    report['cartpole_episode_budget_and_table'] = 'PASS'
    (OUT/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
