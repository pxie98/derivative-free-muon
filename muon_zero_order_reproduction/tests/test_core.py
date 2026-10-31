import numpy as np

from muon_dfo import (
    MatrixRegression,
    direct_structured_search,
    entrywise_dfo_muon,
    full_gradient_muon,
    low_rank_dfo_muon,
    noisy_gradient_muon,
    polar_factor,
)


def small_problem():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(3, 30))
    w_star = rng.normal(size=(2, 3))
    return MatrixRegression(x=x, y=w_star @ x, w_star=w_star), np.zeros_like(w_star)


def test_polar_factor_has_unit_singular_values():
    matrix = np.arange(1.0, 13.0).reshape(3, 4)
    factor = polar_factor(matrix)
    singular_values = np.linalg.svd(factor, compute_uv=False)
    np.testing.assert_allclose(singular_values, np.ones(3), atol=1e-12)


def test_newton_schulz_preserves_shape_and_is_finite():
    matrix = np.arange(1.0, 13.0).reshape(4, 3)
    factor = polar_factor(matrix, method="newton_schulz")
    assert factor.shape == matrix.shape
    assert np.isfinite(factor).all()


def test_ridge_solution_is_stationary():
    problem, _ = small_problem()
    np.testing.assert_allclose(problem.gradient(problem.ridge_solution()), 0.0, atol=1e-11)


def test_entrywise_tracks_full_gradient_for_small_radius():
    problem, w0 = small_problem()
    kwargs = dict(iterations=8, learning_rate=0.02, beta=0.9)
    exact = full_gradient_muon(problem, w0, **kwargs)
    dfo = entrywise_dfo_muon(problem, w0, fd_radius=1e-7, **kwargs)
    np.testing.assert_allclose(dfo.objective, exact.objective, rtol=2e-4, atol=2e-7)
    assert dfo.function_evaluations[-1] == 1 + 8 * (w0.size + 1)


def test_randomized_methods_are_deterministic_and_count_queries():
    problem, w0 = small_problem()
    kwargs = dict(
        iterations=3, learning_rate=0.01, beta=0.9,
        fd_radius=1e-4, probes=2, seed=99,
    )
    first = low_rank_dfo_muon(problem, w0, **kwargs)
    second = low_rank_dfo_muon(problem, w0, **kwargs)
    np.testing.assert_array_equal(first.objective, second.objective)
    assert first.function_evaluations[-1] == 1 + 3 * (2 + 1)

    direct = direct_structured_search(
        problem, w0, iterations=3, learning_rate=0.001,
        fd_radius=1e-4, candidates=4, seed=101,
    )
    assert direct.function_evaluations[-1] == 1 + 3 * (4 + 1)


def test_noisy_gradient_muon_is_deterministic_and_zero_noise_matches_full():
    problem, w0 = small_problem()
    kwargs = dict(iterations=5, learning_rate=0.02, beta=0.9)
    exact = full_gradient_muon(problem, w0, **kwargs)
    zero_noise = noisy_gradient_muon(
        problem, w0, noise_level=0.0, seed=123, **kwargs
    )
    np.testing.assert_array_equal(zero_noise.objective, exact.objective)
    first = noisy_gradient_muon(problem, w0, noise_level=0.5, seed=123, **kwargs)
    second = noisy_gradient_muon(problem, w0, noise_level=0.5, seed=123, **kwargs)
    np.testing.assert_array_equal(first.objective, second.objective)
