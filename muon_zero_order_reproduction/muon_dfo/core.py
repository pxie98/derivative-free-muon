"""Algorithms from *Derivative-Free Structured Updates for Muon*.

The experiment parameters are in config.json and in the current manuscript.
This module separates the mathematical methods from the experiment choices
and counts every black-box objective evaluation explicitly. Saved historical
endpoints can vary under reruns of the fixed-step polar iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class MatrixRegression:
    """Regularized synthetic matrix regression problem."""

    x: Array
    y: Array
    w_star: Array
    regularization: float = 0.0

    def objective(self, w: Array) -> float:
        residual = w @ self.x - self.y
        return float(
            np.sum(residual * residual) / (2.0 * self.x.shape[1])
            + 0.5 * self.regularization * np.sum(w * w)
        )

    def gradient(self, w: Array) -> Array:
        return (w @ self.x - self.y) @ self.x.T / self.x.shape[
            1
        ] + self.regularization * w

    def ridge_solution(self) -> Array:
        n = self.x.shape[0]
        gram = self.x @ self.x.T / self.x.shape[1]
        rhs = self.y @ self.x.T / self.x.shape[1]
        return np.linalg.solve(gram + self.regularization * np.eye(n), rhs.T).T


@dataclass
class RunResult:
    method: str
    objective: Array
    relative_objective: Array
    recovery_error: Array
    function_evaluations: Array
    final_w: Array


def polar_factor(matrix: Array, *, method: str = "svd", steps: int = 5) -> Array:
    """Return a stable semi-orthogonal polar factor.

    ``svd`` is the numerically robust reference. ``newton_schulz`` implements
    the short quintic iteration commonly used by Muon and is included for
    fidelity to the manuscript.  Zero matrices map to zero.
    """

    matrix = np.asarray(matrix, dtype=float)
    norm = float(np.linalg.norm(matrix, ord="fro"))
    if norm <= np.finfo(float).eps:
        return np.zeros_like(matrix)
    if method == "svd":
        u, _, vt = np.linalg.svd(matrix, full_matrices=False)
        return u @ vt
    if method != "newton_schulz":
        raise ValueError(f"unknown polar method: {method}")

    transposed = matrix.shape[0] > matrix.shape[1]
    x = (matrix.T if transposed else matrix) / norm
    # Coefficients used in the public Muon Newton--Schulz implementation.
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    return x.T if transposed else x


def _normalize_rank_one(rng: np.random.Generator, m: int, n: int) -> Array:
    u = rng.normal(size=m)
    v = rng.normal(size=n)
    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)
    return np.outer(u, v)


def _result(
    method: str,
    problem: MatrixRegression,
    history: list[float],
    recovery: list[float],
    evaluations: list[int],
    w: Array,
) -> RunResult:
    optimum = problem.objective(problem.ridge_solution())
    denominator = max(history[0] - optimum, np.finfo(float).eps)
    relative = np.maximum((np.asarray(history) - optimum) / denominator, 0.0)
    return RunResult(
        method=method,
        objective=np.asarray(history),
        relative_objective=relative,
        recovery_error=np.asarray(recovery),
        function_evaluations=np.asarray(evaluations, dtype=int),
        final_w=w.copy(),
    )


def _record(
    problem: MatrixRegression, w: Array, calls: int
) -> tuple[float, float, int]:
    value = problem.objective(w)
    error = np.linalg.norm(w - problem.w_star) / max(
        np.linalg.norm(problem.w_star), np.finfo(float).eps
    )
    return value, float(error), calls + 1


def full_gradient_muon(
    problem: MatrixRegression,
    w0: Array,
    *,
    iterations: int,
    learning_rate: float,
    beta: float,
    polar_method: str = "svd",
) -> RunResult:
    w = np.asarray(w0, dtype=float).copy()
    momentum = np.zeros_like(w)
    history, recovery, evaluations = [], [], []
    calls = 0
    value, error, calls = _record(problem, w, calls)
    history.append(value)
    recovery.append(error)
    evaluations.append(calls)
    for _ in range(iterations):
        gradient = problem.gradient(w)
        momentum = beta * momentum + (1.0 - beta) * gradient
        w -= learning_rate * polar_factor(momentum, method=polar_method)
        value, error, calls = _record(problem, w, calls)
        history.append(value)
        recovery.append(error)
        evaluations.append(calls)
    return _result("Full-gradient Muon", problem, history, recovery, evaluations, w)


def noisy_gradient_muon(
    problem: MatrixRegression,
    w0: Array,
    *,
    iterations: int,
    learning_rate: float,
    beta: float,
    noise_level: float,
    seed: int,
    polar_method: str = "svd",
) -> RunResult:
    """Muon driven by an additive noisy first-order oracle.

    ``noise_level`` is dimensionless.  At every iteration, each gradient entry
    receives independent Gaussian noise whose standard deviation is
    ``noise_level * ||G(W_0)||_F / sqrt(m*n)``.  Fixing the scale at the initial
    gradient makes the signal-to-noise ratio deteriorate near the solution, as
    it does for an oracle with an irreducible gradient-noise floor.
    """

    if noise_level < 0:
        raise ValueError("noise_level must be nonnegative")
    rng = np.random.default_rng(seed)
    w = np.asarray(w0, dtype=float).copy()
    momentum = np.zeros_like(w)
    initial_scale = np.linalg.norm(problem.gradient(w), ord="fro") / np.sqrt(w.size)
    history, recovery, evaluations = [], [], []
    calls = 0
    value, error, calls = _record(problem, w, calls)
    history.append(value)
    recovery.append(error)
    evaluations.append(calls)
    for _ in range(iterations):
        gradient = problem.gradient(w)
        gradient += noise_level * initial_scale * rng.normal(size=w.shape)
        momentum = beta * momentum + (1.0 - beta) * gradient
        w -= learning_rate * polar_factor(momentum, method=polar_method)
        value, error, calls = _record(problem, w, calls)
        history.append(value)
        recovery.append(error)
        evaluations.append(calls)
    return _result(
        f"Noisy-gradient Muon (sigma={noise_level:g})",
        problem,
        history,
        recovery,
        evaluations,
        w,
    )


def entrywise_dfo_muon(
    problem: MatrixRegression,
    w0: Array,
    *,
    iterations: int,
    learning_rate: float,
    beta: float,
    fd_radius: float,
    polar_method: str = "svd",
) -> RunResult:
    """Coordinate forward-difference reconstruction (reference method)."""

    w = np.asarray(w0, dtype=float).copy()
    momentum = np.zeros_like(w)
    history, recovery, evaluations = [], [], []
    calls = 0
    base, error, calls = _record(problem, w, calls)
    history.append(base)
    recovery.append(error)
    evaluations.append(calls)
    for _ in range(iterations):
        gradient = np.empty_like(w)
        for index in np.ndindex(w.shape):
            perturbed = w.copy()
            perturbed[index] += fd_radius
            gradient[index] = (problem.objective(perturbed) - base) / fd_radius
            calls += 1
        momentum = beta * momentum + (1.0 - beta) * gradient
        w -= learning_rate * polar_factor(momentum, method=polar_method)
        base, error, calls = _record(problem, w, calls)
        history.append(base)
        recovery.append(error)
        evaluations.append(calls)
    return _result("Entrywise DFO-Muon", problem, history, recovery, evaluations, w)


def low_rank_dfo_muon(
    problem: MatrixRegression,
    w0: Array,
    *,
    iterations: int,
    learning_rate: float,
    beta: float,
    fd_radius: float,
    probes: int,
    seed: int,
    polar_method: str = "svd",
) -> RunResult:
    """RO-DFO-Muon: random normalized rank-one probing and full momentum."""

    rng = np.random.default_rng(seed)
    w = np.asarray(w0, dtype=float).copy()
    momentum = np.zeros_like(w)
    m, n = w.shape
    history, recovery, evaluations = [], [], []
    calls = 0
    base, error, calls = _record(problem, w, calls)
    history.append(base)
    recovery.append(error)
    evaluations.append(calls)
    for _ in range(iterations):
        surrogate = np.zeros_like(w)
        for _ in range(probes):
            direction = _normalize_rank_one(rng, m, n)
            alpha = (problem.objective(w + fd_radius * direction) - base) / fd_radius
            calls += 1
            surrogate += alpha * direction
        # Isotropic rank-one probes attenuate the expectation by m*n.
        surrogate *= (m * n) / probes
        momentum = beta * momentum + (1.0 - beta) * surrogate
        w -= learning_rate * polar_factor(momentum, method=polar_method)
        base, error, calls = _record(problem, w, calls)
        history.append(base)
        recovery.append(error)
        evaluations.append(calls)
    return _result("Low-rank DFO-Muon", problem, history, recovery, evaluations, w)


def low_rank_dfo_muon_v2(
    problem: MatrixRegression,
    w0: Array,
    *,
    iterations: int,
    learning_rate: float,
    beta: float,
    fd_radius: float,
    polar_method: str = "svd",
) -> RunResult:
    """Exhaustive canonical rank-one probing, scaled by fd_radius (V2)."""
    w = np.asarray(w0, dtype=float).copy()
    momentum = np.zeros_like(w)
    m, n = w.shape
    history, recovery, evaluations = [], [], []
    calls = 0
    base, error, calls = _record(problem, w, calls)
    history.append(base)
    recovery.append(error)
    evaluations.append(calls)
    for _ in range(iterations):
        surrogate = np.zeros_like(w)
        for index in np.ndindex(w.shape):
            delta_perturbed = np.zeros_like(w)
            delta_perturbed[index] += fd_radius
            alpha = (problem.objective(w + delta_perturbed) - base) / fd_radius
            surrogate += alpha * delta_perturbed
            calls += 1
        # For a matrix parameter, this legacy size factor equals one.
        surrogate *= (m * n) / w.size
        momentum = beta * momentum + (1.0 - beta) * surrogate
        w -= learning_rate * polar_factor(momentum, method=polar_method)
        base, error, calls = _record(problem, w, calls)
        history.append(base)
        recovery.append(error)
        evaluations.append(calls)
    return _result("Low-rank DFO-Muon v2", problem, history, recovery, evaluations, w)


def direct_structured_search(
    problem: MatrixRegression,
    w0: Array,
    *,
    iterations: int,
    learning_rate: float,
    fd_radius: float,
    candidates: int,
    seed: int,
) -> RunResult:
    """One-sided random rank-one direct-search control."""

    rng = np.random.default_rng(seed)
    w = np.asarray(w0, dtype=float).copy()
    m, n = w.shape
    history, recovery, evaluations = [], [], []
    calls = 0
    base, error, calls = _record(problem, w, calls)
    history.append(base)
    recovery.append(error)
    evaluations.append(calls)
    for _ in range(iterations):
        best_score = np.inf
        best_direction = None
        for _ in range(candidates):
            direction = _normalize_rank_one(rng, m, n)
            score = (problem.objective(w + fd_radius * direction) - base) / fd_radius
            calls += 1
            if score < best_score:
                best_score, best_direction = score, direction
        # The pseudocode always moves.  Retaining that behavior is important:
        # it explains the weak/noisy direct-search baseline in the paper.
        assert best_direction is not None
        w += learning_rate * best_direction
        base, error, calls = _record(problem, w, calls)
        history.append(base)
        recovery.append(error)
        evaluations.append(calls)
    return _result(
        "Direct structured search", problem, history, recovery, evaluations, w
    )
