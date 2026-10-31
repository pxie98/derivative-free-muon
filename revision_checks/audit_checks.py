"""Validate saved objectives, budgets, seed separation and serial replays."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
RESULTS = ROOT/'results'
sys.path.insert(0, str(PROJECT/'realdata_ct'))
from run_walnut import problem
from ct_baselines import compact_polar, optimize


def nuclear(a):
    return float(np.linalg.svd(a, compute_uv=False).sum())


def tracking_check():
    rng = np.random.default_rng(20261030)
    hessian = np.diag([.4, 1., 2.])
    w = rng.normal(size=(4, 3))
    objective = lambda z: float(np.sum((z@hessian)*z)/2)
    initial = objective(w)
    momentum = np.zeros_like(w)
    errors, gradients, steps = [], [], []
    weighted_gradient, weighted_error, squared_steps = 0., 0., 0.
    maximum_excess = -np.inf
    for t in range(100):
        gradient = w@hessian
        estimate = gradient+.01*rng.normal(size=w.shape)
        momentum = .9*momentum+.1*estimate
        error = nuclear(momentum-gradient)
        if t:
            bound = .9*errors[-1]+.9*2*3*steps[-1]+.1*nuclear(estimate-gradient)
            maximum_excess = max(maximum_excess, error-bound)
            assert error <= bound+1e-12
        eta = .01/np.sqrt(1+t/100)
        weighted_gradient += eta*nuclear(gradient)
        weighted_error += eta*error
        squared_steps += eta*eta
        direction, _ = compact_polar(momentum)
        w -= eta*direction
        errors.append(error)
        gradients.append(gradient.copy())
        steps.append(eta)
    rhs = initial+2*weighted_error+(2*3/2)*squared_steps
    assert weighted_gradient <= rhs+1e-12
    assert objective(w) >= -1e-12
    return dict(iterations=100, tracking_maximum_excess=float(maximum_excess),
                stationarity_left=weighted_gradient/sum(steps),
                stationarity_upper=rhs/sum(steps))


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true', help='repeat seed 901 for all four added methods')
    args = parser.parse_args()
    paths = sorted(p for p in RESULTS.glob('*.json') if p.name.startswith(
        ('tune_', 'test_', 'sensitivity_', 'timing_')) and not p.name.endswith('_environment.json'))
    report = dict(records=0, objective_max_abs_error=0., residual_max_abs_error=0.,
                  budgets='PASS', tuning_test_seeds='PASS', serial_replays=0,
                  original_ct_trajectory_matches=0)
    problems = {v: problem(v) for v in [20, 120]}
    records = []
    for path in paths:
        r = json.loads(path.read_text())
        records.append(r)
        n, a, b, full_a, full_b, _ = problems[r['views']]
        saved = np.load(path.with_suffix('.npz'))
        history, x = saved['history'], saved['w'].ravel(order='F')
        residual = a@x-b
        value = (np.dot(residual, residual)+10*np.dot(x, x))/(2*np.dot(b, b))
        residual_norm = np.linalg.norm(residual)/np.linalg.norm(b)
        delta = abs(value-r['objective'])
        report['objective_max_abs_error'] = max(report['objective_max_abs_error'], float(delta))
        report['residual_max_abs_error'] = max(report['residual_max_abs_error'],
                                               float(abs(residual_norm-r['data_residual'])))
        assert delta < 1e-12*max(1., abs(value)), path.name
        assert abs(residual_norm-r['data_residual']) < 1e-12, path.name
        if r['views'] == 20:
            mask = np.ones(len(full_b), bool)
            mask[(n*np.arange(0, 120, 6)[:, None]+np.arange(n)).ravel()] = False
            heldout = np.linalg.norm((full_a@x-full_b)[mask])/np.linalg.norm(full_b[mask])
            assert abs(heldout-r['heldout_residual']) < 1e-12
        count = n*n if r['method'] == 'coordinate' else (1 if r['method'] == 'jaguar' else r['q'])
        cost = count+1 if r['method'] == 'subspace_muon' else 2*count+1
        expected = 1+((20000-1)//cost)*cost
        assert r['calls'] == expected <= 20000
        assert r['iterations'] == (expected-1)//cost
        np.testing.assert_array_equal(np.diff(history[:, 0]), cost)
        assert history[0, 0] == 1 and history[-1, 0] == r['calls']
        assert history[-1, 2] == r['objective']
        if r['phase'] == 'tune':
            assert r['seed'] in [707, 808]
        elif r['phase'] in {'test', 'sensitivity'}:
            assert r['seed'] in [901, 902, 903, 904, 905]
        if r['phase'] == 'timing':
            matching = [p for p in paths if p.name.startswith(
                f"test_{r['views']}_{r['method']}_901_")]
            if r['method'] in {'sampled_coordinate', 'jaguar', 'subspace_muon', 'rank_matched_sgd'}:
                assert len(matching) == 1
                test = np.load(matching[0].with_suffix('.npz'))
                np.testing.assert_array_equal(saved['w'], test['w'])
                np.testing.assert_array_equal(history[:, :3], test['history'][:, :3])
                report['serial_replays'] += 1
        if r['phase'] == 'sensitivity' and r['q'] == 32 and r['radius'] == 1e-3:
            stem = f"test_20_rankone_{r['seed']}_0p01.npz"
            original = np.load(PROJECT/'realdata_ct/results'/stem)
            np.testing.assert_array_equal(saved['w'], original['w'])
            np.testing.assert_array_equal(history[:, :3], original['history'][:, :3])
            report['original_ct_trajectory_matches'] += 1
        report['records'] += 1
    assert report['records'] == 175, report['records']
    assert report['serial_replays'] == 8
    assert report['original_ct_trajectory_matches'] == 5
    selected = json.loads((RESULTS/'selected_steps.json').read_text())
    for v in [20, 120]:
        for method, eta in selected[str(v)].items():
            rows = [r for r in records if r['phase'] == 'tune' and r['views'] == v and r['method'] == method]
            means = {step: np.mean([r['objective'] for r in rows if r['eta'] == step])
                     for step in sorted({r['eta'] for r in rows})}
            assert eta == min(means, key=means.get)
    if args.replay:
        for v in [20, 120]:
            n, _, _, _, _, objective = problems[v]
            for method, eta in selected[str(v)].items():
                replay = optimize(objective, n, method, 901, eta)
                target = next(p for p in paths if p.name.startswith(f'test_{v}_{method}_901_'))
                saved = np.load(target.with_suffix('.npz'))
                np.testing.assert_array_equal(replay.w, saved['w'])
                np.testing.assert_array_equal(replay.history[:, :3], saved['history'][:, :3])
                print('replay', v, method, 'PASS', flush=True)
        report['fresh_replays'] = 8
    old = PROJECT/'review_revision_backup_20261030/muon_zero_order_reproduction/muon_dfo/core.py'
    if old.exists():
        current = PROJECT/'muon_zero_order_reproduction/muon_dfo/core.py'
        assert strip_docstrings(ast.parse(old.read_text())) == strip_docstrings(ast.parse(current.read_text()))
        report['legacy_core_executable_ast_unchanged'] = True
    report['tracking_bound_check'] = tracking_check()
    report['data_sha256'] = hashlib.sha256((PROJECT/'realdata_ct/data/Data82.mat').read_bytes()).hexdigest()
    report['status'] = 'PASS'
    (RESULTS/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
