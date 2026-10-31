"""Check the V2 scaling identity and fixed-iteration query comparisons.

Original tables and records are read, never overwritten. V2 diagnostics use
both a shared iterate path and independently evolved paths, since differences
on those two paths answer different questions.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS = ROOT/'results'
sys.path.insert(0, str(PROJECT/'muon_zero_order_reproduction'))
from muon_dfo import polar_factor
from run_experiments import make_problem


def coordinate_estimates(problem, w, radius, base):
    coordinate = np.zeros_like(w)
    scaled = np.zeros_like(w)
    for index in np.ndindex(w.shape):
        atom = np.zeros_like(w)
        atom[index] = radius
        slope = (problem.objective(w+atom)-base)/radius
        coordinate[index] = slope
        scaled += slope*atom
    return coordinate, scaled


def v2_check(config, case):
    problem, w = make_problem(case)
    independent = w.copy()
    shared_m = np.zeros_like(w)
    scaled_m = np.zeros_like(w)
    independent_m = np.zeros_like(w)
    beta, h, eta = config['beta'], config['fd_radius'], case['learning_rate']
    rows = []
    threshold_hits = [0, 0]
    gram = problem.x@problem.x.T/problem.x.shape[1]
    fd_bias = .5*h*(np.diag(gram)+problem.regularization)[None, :]
    for t in range(config['iterations']):
        base = problem.objective(w)
        g, scaled_g = coordinate_estimates(problem, w, h, base)
        _, other_g = coordinate_estimates(problem, independent, h, problem.objective(independent))
        shared_m = beta*shared_m+(1-beta)*g
        scaled_m = beta*scaled_m+(1-beta)*scaled_g
        independent_m = beta*independent_m+(1-beta)*other_g
        threshold_hits[0] += int(np.linalg.norm(shared_m) <= np.finfo(float).eps)
        threshold_hits[1] += int(np.linalg.norm(independent_m) <= np.finfo(float).eps)
        d = polar_factor(shared_m)
        shared_scaled_d = polar_factor(scaled_m)
        unscaled_d = polar_factor(scaled_m/h)
        independent_d = polar_factor(independent_m)
        singular = np.linalg.svd(shared_m, compute_uv=False)
        row = dict(iteration=t+1,
                   shared_estimate_relative_error=float(np.linalg.norm(scaled_g/h-g)
                                                       / max(np.linalg.norm(g), 1e-300)),
                   finite_difference_formula_error=float(np.linalg.norm(g-problem.gradient(w)-fd_bias)),
                   shared_momentum_relative_error=float(np.linalg.norm(scaled_m/h-shared_m)
                                                        / max(np.linalg.norm(shared_m), 1e-300)),
                   shared_polar_direction_error=float(np.linalg.norm(shared_scaled_d-d)),
                   unscaled_polar_direction_error=float(np.linalg.norm(unscaled_d-d)),
                   relative_smallest_singular_value=float(singular[-1]/singular[0]),
                   independent_direction_error=float(np.linalg.norm(independent_d-d)))
        w -= eta*d
        independent -= eta*independent_d
        row.update(parameter_distance=float(np.linalg.norm(w-independent)),
                   entry_objective=problem.objective(w), v2_objective=problem.objective(independent))
        rows.append(row)
    name = f"v2_case_{case['id']}"
    with (RESULTS/(name+'.csv')).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    original = json.loads((PROJECT/'muon_zero_order_reproduction/results'/
                           f"matrix_regression_case_{case['id']}_summary.json").read_text())
    terminal = {r['method']: r['final_objective'] for r in original['methods']}
    first = next((r for r in rows if r['parameter_distance'] > 1e-8), None)
    return dict(case=case['id'],
                max_shared_estimate_relative_error=max(r['shared_estimate_relative_error'] for r in rows),
                max_shared_momentum_relative_error=max(r['shared_momentum_relative_error'] for r in rows),
                max_shared_polar_direction_error=max(r['shared_polar_direction_error'] for r in rows),
                max_unscaled_polar_direction_error=max(r['unscaled_polar_direction_error'] for r in rows),
                max_finite_difference_formula_error=max(r['finite_difference_formula_error'] for r in rows),
                min_relative_singular_value=min(r['relative_smallest_singular_value'] for r in rows),
                first_parameter_distance_above_1e_8=first,
                zero_threshold_hits=threshold_hits,
                rerun_terminal_entry=rows[-1]['entry_objective'],
                rerun_terminal_v2=rows[-1]['v2_objective'],
                recorded_terminal_entry=terminal['Entrywise DFO-Muon'],
                recorded_terminal_v2=terminal['Low-rank DFO-Muon v2'])


def query_targets():
    rows = []
    for case in range(1, 5):
        path = PROJECT/'muon_zero_order_reproduction/results'/f'matrix_regression_case_{case}.csv'
        with path.open() as f:
            values = list(csv.DictReader(f))
        for method in dict.fromkeys(r['method'] for r in values):
            selected = [r for r in values if r['method'] == method]
            for threshold in [.1, .01, .001]:
                hit = next((r for r in selected if float(r['relative_objective']) <= threshold), None)
                rows.append(dict(case=case, method=method, relative_gap_target=threshold,
                                 calls_to_target=int(hit['function_evaluations']) if hit else None,
                                 iteration_to_target=int(hit['iteration']) if hit else None,
                                 last_recorded_calls=int(selected[-1]['function_evaluations']),
                                 extra_gradient_work_excluded=method=='Full-gradient Muon'))
    (RESULTS/'query_targets.json').write_text(json.dumps(rows, indent=2)+'\n')
    with (RESULTS/'query_targets.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank_variance():
    rng = np.random.default_rng(314159)
    n, q, trials = 8, 4, 10000
    u = rng.normal(size=(trials, q, n))
    v = rng.normal(size=(trials, q, n))
    u /= np.linalg.norm(u, axis=2, keepdims=True)
    v /= np.linalg.norm(v, axis=2, keepdims=True)
    left = np.linalg.qr(rng.normal(size=(n, n)))[0]
    right = np.linalg.qr(rng.normal(size=(n, n)))[0]
    rows = []
    for rank in [1, 2, 8]:
        gradient = left[:, :rank]@right[:, :rank].T/np.sqrt(rank)
        slopes = np.einsum('bqi,ij,bqj->bq', u, gradient, v)
        estimate = (n*n/q)*np.einsum('bq,bqi,bqj->bij', slopes, u, v)
        errors = np.sum((estimate-gradient)**2, axis=(1, 2))
        rows.append(dict(rank=rank, frobenius_norm=float(np.linalg.norm(gradient)),
                         empirical_mse=float(errors.mean()), standard_error=float(errors.std(ddof=1)/np.sqrt(trials)),
                         theoretical_mse=(n*n-1)/q, q=q, trials=trials, seed=314159))
    (RESULTS/'rank_variance.json').write_text(json.dumps(rows, indent=2)+'\n')


def main():
    RESULTS.mkdir(exist_ok=True)
    config = json.loads((PROJECT/'muon_zero_order_reproduction/config.json').read_text())
    report = [v2_check(config, case) for case in config['cases']]
    (RESULTS/'v2_diagnostics.json').write_text(json.dumps(report, indent=2)+'\n')
    query_targets()
    rank_variance()
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
