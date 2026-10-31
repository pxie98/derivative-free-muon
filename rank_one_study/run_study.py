"""Repeat four fixed regression problems with 30 fresh probe seeds each."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
from importlib.metadata import version
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'muon_zero_order_reproduction'))
from run_experiments import make_problem
from muon_dfo import entrywise_dfo_muon
from rank_one_muon import rank_one_muon

SEEDS = list(range(2101, 2131))
TARGETS = [0.5, 0.1, 0.01]


def first_hit(gaps, calls, target):
    indices = np.flatnonzero(np.asarray(gaps) <= target)
    return int(calls[indices[0]]) if indices.size else None


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    environment = dict(python=sys.version, numpy=np.__version__,
                       matplotlib=version('matplotlib'),
                       platform=platform.platform(), machine=platform.machine(),
                       blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                       omp_threads=os.environ.get('OMP_NUM_THREADS'),
                       polar='numpy.linalg.svd(full_matrices=False); economy completion',
                       zero_threshold=float(np.finfo(float).eps))
    (out/'environment.json').write_text(json.dumps(environment, indent=2)+'\n')
    cfg = json.loads((ROOT/'muon_zero_order_reproduction/config.json').read_text())
    rows, summaries = [], []
    for case in cfg['cases']:
        problem, w0 = make_problem(case)
        optimum = problem.objective(problem.ridge_solution())
        denominator = problem.objective(w0) - optimum
        common = dict(iterations=cfg['iterations'], learning_rate=case['learning_rate'],
                      beta=cfg['beta'], fd_radius=cfg['fd_radius'])
        coordinate = entrywise_dfo_muon(problem, w0, **common)
        gaps = (coordinate.objective-optimum)/denominator
        for t, (gap, call) in enumerate(zip(gaps, coordinate.function_evaluations)):
            rows.append([case['id'], 'Coordinate', -1, t, int(call), float(gap)])
        all_gaps, all_calls = [], []
        for seed in SEEDS:
            run = rank_one_muon(problem.objective, w0, **common,
                                probes=case['low_rank_probes'], seed=seed)
            normalized = (run.objective-optimum)/denominator
            all_gaps.append(normalized)
            all_calls.append(run.calls)
            for t, (gap, call) in enumerate(zip(normalized, run.calls)):
                rows.append([case['id'], 'RO-DFO-Muon', seed, t, int(call), float(gap)])
        targets = []
        for target in TARGETS:
            hits = [first_hit(g, c, target) for g, c in zip(all_gaps, all_calls)]
            successful = [h for h in hits if h is not None]
            coord_hit = first_hit(gaps, coordinate.function_evaluations, target)
            median = float(np.median(successful)) if successful else None
            targets.append(dict(target=target, coordinate_calls=coord_hit,
                                successes=len(successful), trials=len(SEEDS),
                                median_calls_successful=median,
                                min_calls=min(successful) if successful else None,
                                max_calls=max(successful) if successful else None,
                                ratio=coord_hit/median if median and coord_hit else None))
        summaries.append(dict(case=case['id'], targets=targets,
                              terminal_gap_median=float(np.median(np.array(all_gaps)[:, -1]))))
        print('completed case', case['id'], targets, flush=True)
    with (out/'trajectories.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case', 'method', 'seed', 'iteration', 'calls', 'relative_gap'])
        writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(dict(
        protocol='Fixed original problems and parameters; 200 updates; no new tuning.',
        seeds=SEEDS, targets=TARGETS, cases=summaries), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    main(parser.parse_args().output)
