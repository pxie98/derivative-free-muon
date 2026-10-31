"""Rank-One Derivative-Free Muon for a scalar matrix objective.

This is the forward-probe algorithm previously called low_rank_dfo_muon,
with a general callable interface and an optional central difference.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass
class Result:
    matrix: np.ndarray
    objective: np.ndarray
    calls: np.ndarray


def rank_one_muon(objective: Callable, initial: np.ndarray, *, iterations: int,
                  learning_rate: float, beta: float = 0.9,
                  fd_radius: float = 1e-4, probes: int = 4,
                  seed: int = 0, difference: str = "forward") -> Result:
    """Minimize using normalized Gaussian rank-one probes and full momentum.

    Every initial, perturbed, and endpoint evaluation is counted. The SVD
    includes the economy-size null-space completion used in the regression
    experiments. No stopping rule, step tuning, or gradient access is used.
    """
    if (iterations < 0 or probes < 1 or fd_radius <= 0 or learning_rate <= 0
            or not 0 <= beta < 1 or difference not in {"forward", "central"}):
        raise ValueError("Invalid algorithm parameters")
    w = np.asarray(initial, dtype=float).copy()
    if w.ndim != 2 or not np.isfinite(w).all():
        raise ValueError("initial must be a finite matrix")
    rng = np.random.default_rng(seed)
    m, n = w.shape
    momentum = np.zeros_like(w)
    base = float(objective(w))
    values, calls = [base], [1]
    count = 1
    for _ in range(iterations):
        estimate = np.zeros_like(w)
        for _ in range(probes):
            u, v = rng.normal(size=m), rng.normal(size=n)
            u /= max(np.linalg.norm(u), np.finfo(float).eps)
            v /= max(np.linalg.norm(v), np.finfo(float).eps)
            atom = np.outer(u, v)
            plus = float(objective(w + fd_radius * atom))
            count += 1
            if difference == "central":
                minus = float(objective(w - fd_radius * atom))
                count += 1
                slope = (plus - minus) / (2 * fd_radius)
            else:
                slope = (plus - base) / fd_radius
            estimate += slope * atom
        estimate *= (m * n) / probes
        momentum = beta * momentum + (1 - beta) * estimate
        if np.linalg.norm(momentum) > np.finfo(float).eps:
            u, _, vt = np.linalg.svd(momentum, full_matrices=False)
            w -= learning_rate * (u @ vt)
        base = float(objective(w))
        count += 1
        values.append(base)
        calls.append(count)
    return Result(w, np.asarray(values), np.asarray(calls))
