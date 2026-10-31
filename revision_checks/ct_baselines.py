"""Additional function-value baselines for the measured walnut problem.

Every query evaluates the supplied objective. No adjoint, quadratic shortcut,
or reference solution is available to optimize(). The published methods are
adapted to exact compact SVD factors and the common experimental schedule;
these adaptations are recorded in the reproduction notes.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass
class Result:
    w: np.ndarray
    history: np.ndarray
    times: dict[str, float]


def compact_polar(matrix):
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    keep = s > np.finfo(float).eps * max(matrix.shape) * s[0]
    return u[:, keep] @ vt[keep], int(keep.sum())


def rank_matched_direction(matrix):
    s = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.count_nonzero(s > np.finfo(float).eps * max(matrix.shape) * s[0]))
    norm = np.linalg.norm(matrix)
    return np.sqrt(rank) * matrix / norm if norm else matrix.copy()


def optimize(objective, n, method, seed, eta0, *, budget=20000, q=32,
             radius=1e-3, beta=.9, projection_rank=8, refresh=100):
    """Return iterates and separately counted objective/update/probe timings.

    sampled_coordinate: q fresh canonical atoms, full momentum decay.
    jaguar: one canonical atom, momentum refresh only at its sampled entry
        (Petrov et al., arXiv:2506.04430v4, Section 3.3).
    subspace_muon: Gaussian directions in a periodically resampled left
        subspace, without momentum (Lang et al., arXiv:2602.17155v2,
        equations (10)--(11)); compact SVD replaces Newton--Schulz.
    rank_matched_sgd: same probes and momentum as rankone, with instantaneous
        Frobenius norm equal to that of its compact polar factor.
    The five remaining names reproduce the original CT estimators.
    """
    allowed = {'rankone', 'orthogonal', 'sphere', 'zo_sgd', 'coordinate',
               'sampled_coordinate', 'jaguar', 'subspace_muon', 'rank_matched_sgd'}
    if method not in allowed:
        raise ValueError(method)
    if n < 1 or q < 1 or radius <= 0 or eta0 <= 0 or not 0 <= beta < 1:
        raise ValueError('invalid dimensions or optimizer parameters')
    if method == 'subspace_muon' and not 1 <= projection_rank <= n:
        raise ValueError('projection_rank must be between 1 and n')
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n))
    momentum = w.copy()
    calls = 0
    times = dict(objective=0., probes_and_estimate=0., update=0.)

    def query(z):
        nonlocal calls
        clock = time.perf_counter()
        value = objective(z)
        times['objective'] += time.perf_counter() - clock
        calls += 1
        return value

    start = time.perf_counter()
    f = query(w)
    history = [[calls, 0, f, 0.]]
    count = n*n if method == 'coordinate' else (1 if method == 'jaguar' else q)
    if method in {'sampled_coordinate', 'orthogonal'} and count > n*n:
        raise ValueError('sampling without replacement requires q <= n*n')
    cost = count+1 if method == 'subspace_muon' else 2*count+1
    t = 0
    projection = None
    while calls + cost <= budget:
        clock = time.perf_counter()
        previous_objective_seconds = times['objective']
        if method == 'subspace_muon':
            if projection is None or t % refresh == 0:
                projection = np.linalg.qr(rng.normal(size=(n, projection_rank)))[0]
            estimate = np.zeros((projection_rank, n))
            for _ in range(count):
                psi = rng.normal(size=estimate.shape)
                slope = (query(w + radius * (projection @ psi)) - f) / radius
                estimate += slope * psi
            estimate /= count
        elif method == 'jaguar':
            i, j = rng.integers(n), rng.integers(n)
            atom = np.zeros_like(w)
            atom[i, j] = 1.
            slope = (query(w+radius*atom)-query(w-radius*atom))/(2*radius)
        else:
            estimate = np.zeros_like(w)
            if method == 'orthogonal':
                left = np.linalg.qr(rng.normal(size=(n, n)))[0]
                right = np.linalg.qr(rng.normal(size=(n, n)))[0]
                atoms = rng.choice(n*n, count, replace=False)
            elif method == 'sampled_coordinate':
                atoms = rng.choice(n*n, count, replace=False)
            for k in range(count):
                if method in {'coordinate', 'sampled_coordinate'}:
                    atom = np.zeros_like(w)
                    atom.flat[k if method == 'coordinate' else atoms[k]] = 1.
                elif method == 'orthogonal':
                    i, j = divmod(int(atoms[k]), n)
                    atom = np.outer(left[:, i], right[:, j])
                elif method == 'sphere':
                    atom = rng.normal(size=w.shape)
                    atom /= np.linalg.norm(atom)
                else:
                    u, v = rng.normal(size=n), rng.normal(size=n)
                    atom = np.outer(u/np.linalg.norm(u), v/np.linalg.norm(v))
                slope = (query(w+radius*atom)-query(w-radius*atom))/(2*radius)
                estimate += slope*atom
            estimate *= n*n/count
        times['probes_and_estimate'] += (time.perf_counter()-clock
                                        - times['objective']+previous_objective_seconds)
        clock = time.perf_counter()
        if method == 'jaguar':
            momentum[i, j] = beta*momentum[i, j] + (1-beta)*slope
            direction, _ = compact_polar(momentum)
        elif method == 'subspace_muon':
            factor, _ = compact_polar(estimate)
            direction = projection @ factor
        else:
            momentum = beta*momentum + (1-beta)*estimate
            if method == 'zo_sgd':
                norm = np.linalg.norm(momentum)
                direction = np.sqrt(n)*momentum/norm if norm else momentum.copy()
            elif method == 'rank_matched_sgd':
                direction = rank_matched_direction(momentum)
            else:
                direction, _ = compact_polar(momentum)
        eta = eta0/np.sqrt(1+t/100)
        w -= eta*direction
        times['update'] += time.perf_counter()-clock
        f = query(w)
        t += 1
        history.append([calls, t, f, time.perf_counter()-start])
    times['total'] = time.perf_counter()-start
    times['other'] = times['total'] - sum(times[x] for x in
                                        ['objective', 'probes_and_estimate', 'update'])
    return Result(w=w, history=np.asarray(history), times=times)
