"""Run the limited additional checks accompanying the manuscript revision.

Use --phase tune, test, sensitivity, timing, or diagnostics. Tuning and testing
have disjoint seeds. Parallel runs are never used for runtime comparisons.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT/'realdata_ct'))
from run_walnut import problem
from ct_baselines import optimize

RESULTS = ROOT/'results'
METHODS = ['sampled_coordinate', 'jaguar', 'subspace_muon', 'rank_matched_sgd']
GRID = [.001, .003, .01, .03, .1]
JAGUAR_GRID = [1e-5, 3e-5, 1e-4, 3e-4] + GRID
TUNE_SEEDS = [707, 808]
TEST_SEEDS = [901, 902, 903, 904, 905]
BUDGET = 20000


def run_job(job):
    views, method, seed, eta, phase, q, radius = job
    stem = f'{phase}_{views}_{method}_{seed}_{eta:g}_q{q}_h{radius:g}'.replace('.', 'p')
    target = RESULTS/stem
    if target.with_suffix('.json').exists():
        return json.loads(target.with_suffix('.json').read_text())
    n, a, b, full_a, full_b, objective = problem(views)
    result = optimize(objective, n, method, seed, eta, budget=BUDGET, q=q, radius=radius)
    x = result.w.ravel(order='F')
    reference = json.loads((PROJECT/'realdata_ct/results'/f'reference_{views}.json').read_text())
    record = dict(views=views, method=method, seed=seed, eta=eta, phase=phase,
                  q=q, probes_per_update=(1 if method == 'jaguar' else q),
                  radius=radius, beta=.9, projection_rank=8,
                  projection_refresh=100, calls=int(result.history[-1, 0]),
                  iterations=int(result.history[-1, 1]), objective=float(result.history[-1, 2]),
                  gap=float((result.history[-1, 2]-reference['objective'])/(.5-reference['objective'])),
                  data_residual=float(np.linalg.norm(a@x-b)/np.linalg.norm(b)),
                  times=result.times)
    if views == 20:
        mask = np.ones(len(full_b), bool)
        mask[(n*np.arange(0, 120, 6)[:, None]+np.arange(n)).ravel()] = False
        record['heldout_residual'] = float(np.linalg.norm((full_a@x-full_b)[mask])
                                           / np.linalg.norm(full_b[mask]))
    np.savez_compressed(target.with_suffix('.npz'), w=result.w, history=result.history)
    target.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['tune', 'test', 'timing', 'sensitivity', 'diagnostics'],
                        required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.phase == 'diagnostics':
        from numerical_diagnostics import main as diagnostics
        diagnostics()
        return
    if args.phase == 'tune':
        jobs = [(v, m, s, eta, 'tune', 32, 1e-3) for v in [20, 120]
                for m in METHODS for eta in (JAGUAR_GRID if m == 'jaguar' else GRID)
                for s in TUNE_SEEDS]
    elif args.phase == 'sensitivity':
        # One-factor checks; the q=32, h=1e-3 condition is shared once.
        settings = [(q, 1e-3) for q in [8, 32, 64]] + [(32, h) for h in [1e-4, 1e-2]]
        jobs = [(20, 'rankone', s, .01, 'sensitivity', q, h)
                for q, h in settings for s in TEST_SEEDS]
    else:
        selected = json.loads((RESULTS/'selected_steps.json').read_text())
        methods = METHODS + (['rankone', 'zo_sgd', 'coordinate'] if args.phase == 'timing' else [])
        original = json.loads((PROJECT/'realdata_ct/results/selected_steps.json').read_text())
        jobs = [(v, m, s, (selected[str(v)][m] if m in METHODS else original[str(v)][m]),
                 args.phase, 32, 1e-3) for v in [20, 120] for m in methods
                for s in ([901] if args.phase == 'timing' else TEST_SEEDS)]
    workers = 1 if args.phase == 'timing' else args.workers
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(run_job, jobs):
            records.append(record)
            print(args.phase, record['views'], record['method'], record['seed'],
                  record['eta'], f"gap={record['gap']:.6f}", flush=True)
    if args.phase == 'tune':
        selected = {}
        for v in [20, 120]:
            selected[str(v)] = {}
            for m in METHODS:
                means = {eta: np.mean([r['objective'] for r in records
                                       if r['views'] == v and r['method'] == m and r['eta'] == eta])
                         for eta in (JAGUAR_GRID if m == 'jaguar' else GRID)}
                selected[str(v)][m] = min(means, key=means.get)
        (RESULTS/'selected_steps.json').write_text(json.dumps(selected, indent=2)+'\n')
        print('Selected steps:', selected, flush=True)
    environment = dict(python=platform.python_version(), numpy=np.__version__,
                       scipy=scipy.__version__, platform=platform.platform(),
                       machine=platform.machine(), workers=workers,
                       blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
                       budget=BUDGET, step_grid=GRID, jaguar_step_grid=JAGUAR_GRID,
                       tune_seeds=TUNE_SEEDS,
                       evaluation_seeds=TEST_SEEDS,
                       data_sha256=hashlib.sha256((PROJECT/'realdata_ct/data/Data82.mat').read_bytes()).hexdigest())
    (RESULTS/(args.phase+'_environment.json')).write_text(json.dumps(environment, indent=2)+'\n')


if __name__ == '__main__':
    main()
