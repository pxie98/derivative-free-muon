"""Independent numerical checks of constants and boundary cases in the paper.

These checks supplement the derivations; they are not formal proof certificates.
They do not change any optimizer or saved experiment record.
"""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'muon_zero_order_reproduction'))
sys.path.insert(0, str(ROOT/'revision_checks'))
sys.path.insert(0, str(ROOT/'realdata_ct'))
from muon_dfo.core import polar_factor
from ct_baselines import compact_polar
from run_walnut import problem


def nuclear(a):
    return float(np.linalg.svd(a, compute_uv=False).sum())


def finite_difference_bounds():
    # Quadratic forward bias and adversarial bounded oracle errors attain
    # the stated coefficients Lh/2 + 2 delta/h and delta/h, respectively.
    x = .7
    for h in [.03, .2, .8]:
        for delta in [0., 1e-7, .01]:
            plus = .5*(x+h)**2+delta
            base = .5*x*x-delta
            minus = .5*(x-h)**2-delta
            forward = (plus-base)/h
            central = (plus-minus)/(2*h)
            np.testing.assert_allclose(forward-x, h/2+2*delta/h,
                                       rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(central-x, delta/h,
                                       rtol=1e-10, atol=1e-14)
    # f=sin has globally 1-Lipschitz Hessian in one dimension.
    for x in [0., .7, 1.4]:
        for h in [.01, .1, .5]:
            central = (np.sin(x+h)-np.sin(x-h))/(2*h)
            assert abs(central-np.cos(x)) <= h*h/6+1e-14
    return dict(status='PASS', forward_oracle_coefficient=2,
                central_oracle_coefficient=1, central_cubic_coefficient=1/6)


def polar_deficit():
    matrices = [np.diag([1., 1e-18, 0.]), np.zeros((3, 3)),
                np.diag([1e-18, 2e-18, 0.]),
                np.random.default_rng(31).normal(size=(3, 5))]
    max_error = 0.
    for matrix in matrices:
        u, sigma, vt = np.linalg.svd(matrix, full_matrices=False)
        keep = sigma > np.finfo(float).eps*max(matrix.shape)*sigma[0]
        compact, _ = compact_polar(matrix)
        deficit = nuclear(matrix)-float(np.sum(matrix*compact))
        expected = float(sigma[~keep].sum())
        max_error = max(max_error, abs(deficit-expected))
        assert abs(deficit-expected) <= 1e-13*max(1., nuclear(matrix))
        assert np.linalg.norm(compact, 2) <= 1+1e-14
        gradient = np.random.default_rng(72).normal(size=matrix.shape)
        lhs = float(np.sum(gradient*compact))
        rhs = nuclear(gradient)-2*nuclear(matrix-gradient)-deficit
        assert lhs >= rhs-1e-12
    matrix = np.diag([2., .25])
    truncated = np.diag([1., 0.])
    exact_loss = nuclear(matrix)-float(np.sum(matrix*truncated))
    assert exact_loss == .25
    assert float(np.sum(matrix*truncated)) < nuclear(matrix)
    assert float(np.sum(matrix*truncated)) == nuclear(matrix)-exact_loss
    # With G=M, dropping a nonzero singular value violates exact-polar
    # equality; the discarded-value sum is the missing term.
    matrix = np.diag([1., 1e-18])
    factor, _ = compact_polar(matrix)
    assert factor[1, 1] == 0
    assert float(np.sum(matrix*factor)) == 1.
    return dict(status='PASS', deficit_identity_max_abs_error=max_error)


def weighted_descent():
    rng = np.random.default_rng(85)
    target = rng.normal(size=(3, 5))
    weights = np.arange(1., 6.)
    objective = lambda w: float(np.sum(weights*(w-target)**2)/2)
    gradient = lambda w: weights*(w-target)
    w, momentum = np.zeros_like(target), np.zeros_like(target)
    initial = objective(w)
    lhs = error = deficit_sum = step2 = 0.
    for t in range(80):
        g = gradient(w)
        momentum = .9*momentum+.1*g
        u, sigma, vt = np.linalg.svd(momentum, full_matrices=False)
        # Deliberately retain only one mode; occasionally take a zero step.
        direction = np.outer(u[:, 0], vt[0]) if t % 17 else np.zeros_like(w)
        eta = .01/np.sqrt(1+t/10)
        gap = nuclear(momentum)-float(np.sum(momentum*direction))
        lhs += eta*nuclear(g)
        error += eta*nuclear(momentum-g)
        deficit_sum += eta*gap
        step2 += eta*eta
        w -= eta*direction
    rhs = initial+2*error+deficit_sum+5*3*step2/2
    assert lhs <= rhs+1e-11
    return dict(status='PASS', weighted_left=lhs, weighted_upper=rhs,
                iterations=80)


def absolute_zero_cutoff():
    # The actual regression helper returns zero below machine epsilon.
    # Treating that result as an exact polar maximizer would make the
    # old exact-polar stationarity inequality false on this trajectory.
    scale = np.finfo(float).eps/8
    matrix = np.array([[-scale]])
    direction = polar_factor(matrix)
    np.testing.assert_array_equal(direction, np.zeros((1, 1)))
    iterations, eta = 100, .01
    initial = scale/2             # f(w)=scale*(w-1)^2/2, w=0
    lhs = iterations*eta*scale
    old_rhs = initial+scale*iterations*eta*eta/2
    deficit = nuclear(matrix)
    corrected_rhs = old_rhs+iterations*eta*deficit
    assert lhs > old_rhs
    assert lhs <= corrected_rhs
    return dict(status='PASS', old_bound_applicable_to_cutoff=False,
                left_over_scale=lhs/scale,
                exact_map_upper_over_scale=old_rhs/scale,
                corrected_upper_over_scale=corrected_rhs/scale)


def scale_invariance():
    # Variational optimality alone does not enforce a scale-invariant
    # null-space completion. Both choices below are exact maximizers.
    matrix = np.diag([1., 0.])
    def selection(a):
        return np.diag([1., 1. if np.linalg.norm(a) >= 1 else -1.])
    for a in [matrix, .5*matrix]:
        chosen = selection(a)
        assert np.linalg.norm(chosen, 2) == 1
        assert float(np.sum(a*chosen)) == nuclear(a)
    assert not np.array_equal(selection(matrix), selection(.5*matrix))
    compact1, _ = compact_polar(matrix)
    compact2, _ = compact_polar(.5*matrix)
    np.testing.assert_array_equal(compact1, compact2)
    return dict(status='PASS', arbitrary_maximizer_need_not_be_scale_invariant=True,
                compact_factor_is_scale_invariant=True)


def cg_reference_bound():
    rng = np.random.default_rng(91)
    for a in [np.zeros((7, 6)), rng.normal(size=(7, 6))]:
        b, alpha = rng.normal(size=7), 10.
        optimum = np.linalg.solve(a.T@a+alpha*np.eye(6), a.T@b)
        trial = optimum+.02*rng.normal(size=6)
        value = lambda x: (np.linalg.norm(a@x-b)**2+alpha*np.linalg.norm(x)**2)/(2*np.dot(b, b))
        residual = a.T@(a@trial-b)+alpha*trial
        bound = np.dot(residual, residual)/(2*alpha*np.dot(b, b))
        assert -1e-14 <= value(trial)-value(optimum) <= bound+1e-14
    results = {}
    for views in [20, 120]:
        n, a, b, _, _, objective = problem(views)
        saved = np.load(ROOT/f'realdata_ct/results/reference_{views}.npz')['w']
        x = saved.ravel(order='F')
        residual = a.T@(a@x-b)+10*x
        bound = float(np.dot(residual, residual)/(20*np.dot(b, b)))
        record = json.loads((ROOT/f'realdata_ct/results/reference_{views}.json').read_text())
        np.testing.assert_allclose(bound, record['objective_error_bound'], rtol=1e-3, atol=0)
        assert bound < 1e-18
        assert abs(objective(saved)-record['objective']) < 1e-15
        results[str(views)] = bound
    return dict(status='PASS', measured_reference_error_bounds=results)


def main():
    checks = [finite_difference_bounds, polar_deficit, weighted_descent,
              absolute_zero_cutoff, scale_invariance, cg_reference_bound]
    report = {check.__name__: check() for check in checks}
    report.update(status='PASS', checks=len(checks),
                  note='Numerical regression checks supplement the mathematical derivations.')
    out = Path(__file__).parent/'results'
    out.mkdir(exist_ok=True)
    (out/'mathematics_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
