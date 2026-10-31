# Rank-One Derivative-Free Muon

`rank_one_muon.py` provides the paper's main algorithm through a general scalar-objective interface. It is the forward random-probe method formerly labeled “low-rank DFO-Muon”; the numerical method has not been replaced by a new, unvalidated variant. The optional central-difference setting uses twice as many perturbed values. The measured CT driver uses its own compact-SVD implementation and step schedule, documented in `../realdata_ct/`.

```python
import numpy as np
from rank_one_study.rank_one_muon import rank_one_muon

result = rank_one_muon(
    lambda w: float(np.sum((w - 1)**2)), np.zeros((8, 8)),
    iterations=200, learning_rate=.035, probes=4, seed=2101,
)
print(result.objective[-1], result.calls[-1])
```

## Repeat the added study

From the project root, using the pinned dependencies:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 rank_one_study/run_study.py
MPLCONFIGDIR=/tmp/muon_revision_matplotlib python3 rank_one_study/plot_results.py
python3 rank_one_study/audit_study.py
python3 -m pytest -q rank_one_study/test_rank_one.py
```

To keep the supplied records intact, use `run_study.py --output /tmp/ro_muon_replay`. The runner recomputes all records rather than resuming them. The plotting script reads the supplied `results/` directory.

## Protocol and interpretation

The four data problems, their seeds, step sizes, momentum, radius, probe counts, and 200-update horizon are unchanged from `../muon_zero_order_reproduction/config.json`. Only probe seeds vary: 2101–2130 for every problem. There are 120 random-probe runs and four deterministic coordinate references. No step size was retuned. This measures probe variability on four fixed problems, not variability across newly sampled data problems. Targets 0.5, 0.1, and 0.01 are all reported. They were selected with the original archived trajectories available; this is a follow-up validation, not a claim of preregistration.

`environment.json` records Python/NumPy/Matplotlib versions, the platform, thread settings, and the SVD convention. Economy-SVD null-space completion and fixed-step terminal values can be sensitive to the numerical environment. `trajectories.csv` records every iterate's call count and relative gap. `summary.json` records success counts, first-hit call counts, and the ratio of the coordinate count to the median successful random-probe count. A first hit need not mean the method remains below the threshold. Every initial, perturbed, and endpoint call is counted. Neither optimizer uses analytic gradients. The ridge solution is used only for reporting the relative gap.

At target 0.1 all 30 probe seeds succeed per problem, with median count ratios of 6.59, 6.43, 8.27, and 9.94. At target 0.01 no random-probe run succeeds within 200 updates; all coordinate references succeed. Thus the demonstrated advantage is against full coordinate recovery at moderate accuracy under the stated protocol. It does not establish superiority over normalized ZO-SGD, sampled coordinates, or all existing zeroth-order Muon algorithms.

The new Figure 1 shows medians and 10th–90th percentile bands from these runs. Figure 4 is reduced to a single episode-axis plot from the original CartPole CSV. That replot introduces no new CartPole data; its missing experiment driver remains a limitation.
