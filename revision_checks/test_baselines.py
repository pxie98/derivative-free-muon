"""Checks for oracle budgets, adapted baselines, and step normalization."""
from pathlib import Path
import sys

import numpy as np

from ct_baselines import compact_polar, optimize, rank_matched_direction

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'realdata_ct'))
from run_walnut import optimize as original_optimize


def test_original_ct_trajectories_are_preserved():
    target = np.random.default_rng(41).normal(size=(6, 6))
    objective = lambda w: float(np.sum((w-target)**2)/2)
    for method in ['rankone', 'orthogonal', 'sphere', 'zo_sgd', 'coordinate']:
        original_w, history, _ = original_optimize(objective, 6, method, 901, .003, budget=196)
        result = optimize(objective, 6, method, 901, .003, budget=196)
        np.testing.assert_array_equal(result.w, original_w)
        np.testing.assert_array_equal(result.history[:, :3], history[:, :3])


def test_oracle_count_includes_initial_and_endpoint_evaluations():
    for method, q, cost in [('sampled_coordinate', 2, 5), ('jaguar', 2, 3),
                            ('subspace_muon', 3, 4), ('rank_matched_sgd', 2, 5)]:
        count = [0]
        def objective(w):
            count[0] += 1
            return float(np.sum((w-1)**2)/2)
        result = optimize(objective, 4, method, 21, .01, budget=1+3*cost,
                          q=q, projection_rank=2)
        assert count[0] == 1+3*cost == result.history[-1, 0]
        assert result.history[-1, 1] == 3
        assert np.isfinite(result.w).all()


def test_rank_matched_step_matches_compact_polar_norm():
    for matrix in [np.zeros((4, 4)), np.diag([2., .1, 0., 0.]),
                   np.random.default_rng(9).normal(size=(4, 4))]:
        factor, rank = compact_polar(matrix)
        direction = rank_matched_direction(matrix)
        np.testing.assert_allclose(np.linalg.norm(factor), np.sqrt(rank), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(direction), np.linalg.norm(factor), atol=1e-12)


def test_jaguar_first_step_matches_sampled_coordinate_polar_step():
    target = np.arange(1., 17.).reshape(4, 4)
    seed, eta = 7, .02
    rng = np.random.default_rng(seed)
    i, j = rng.integers(4), rng.integers(4)
    expected = np.zeros_like(target)
    expected[i, j] = eta
    result = optimize(lambda w: float(np.sum((w-target)**2)/2), 4, 'jaguar',
                      seed, eta, budget=4)
    np.testing.assert_allclose(result.w, expected, atol=1e-12)


def test_subspace_zero_gradient_has_no_arbitrary_completion():
    result = optimize(lambda w: 3., 4, 'subspace_muon', 99, .03,
                      budget=13, q=3, projection_rank=2, refresh=1)
    np.testing.assert_array_equal(result.w, np.zeros((4, 4)))
    assert result.history[-1, 0] == 13
