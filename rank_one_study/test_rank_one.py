import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'muon_zero_order_reproduction'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from muon_dfo import MatrixRegression, low_rank_dfo_muon
from rank_one_muon import rank_one_muon
from run_study import first_hit


def test_legacy_algorithm_agreement():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(3, 20))
    star = rng.normal(size=(2, 3))
    problem = MatrixRegression(x, star @ x, star, 1e-4)
    kwargs = dict(iterations=12, learning_rate=.03, beta=.9,
                  fd_radius=1e-4, probes=4, seed=31)
    old = low_rank_dfo_muon(problem, np.zeros_like(star), **kwargs)
    new = rank_one_muon(problem.objective, np.zeros_like(star), **kwargs)
    np.testing.assert_allclose(new.matrix, old.final_w, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(new.objective, old.objective, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(new.calls, old.function_evaluations)


def test_actual_oracle_calls_and_first_hit():
    for difference, per_probe in [('forward', 1), ('central', 2)]:
        actual = []
        def objective(w):
            actual.append(w.copy())
            return float(np.sum((w-1)**2))
        run = rank_one_muon(objective, np.zeros((2, 2)), iterations=7,
                            learning_rate=.02, probes=3, difference=difference)
        assert len(actual) == run.calls[-1] == 1 + 7 * (3*per_probe + 1)
    assert first_hit([1, .5, .1], [1, 6, 11], .5) == 6
    assert first_hit([1, .5, .1], [1, 6, 11], .01) is None
