# Additional numerical checks

These files accompany the October 31, 2026 revision of *Derivative-Free Muon for Matrix Optimization with Rank-One Perturbations*. Original experiment records remain in `../muon_zero_order_reproduction/results/` and `../realdata_ct/results/`. This directory stores the additional controls and diagnostics separately.

## Run

Use Python 3.12 and the versions in the project-level `requirements.txt`. Run from the project root:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/muon_revision_matplotlib
python3 revision_checks/run_checks.py --phase diagnostics
python3 revision_checks/run_checks.py --phase tune --workers 2
python3 revision_checks/run_checks.py --phase test --workers 2
python3 revision_checks/run_checks.py --phase sensitivity --workers 2
python3 revision_checks/run_checks.py --phase timing --workers 1
python3 revision_checks/report_checks.py
python3 revision_checks/audit_checks.py --replay
python3 -m pytest -q revision_checks/test_baselines.py
```

The scripts resume saved jobs. For a fresh run, use a separate checkout or move `revision_checks/results/` aside. Timing runs are serial and should be run after the other phases finish. The supplied serial records use seed 901; they are individual reruns, not timing distributions. Environment records distinguish these runs from the older CT timings.

## Added CT controls

All optimizers receive a scalar objective callable. Every objective call applies the measured forward operator; no adjoint or quadratic shortcut is used inside an optimizer. The common ceiling is 20,000 calls, including initialization and endpoint evaluations.

| Control | Measurements and update | Calls in a completed run |
| --- | --- | ---: |
| Sampled-coordinate Muon | 32 distinct coordinate atoms, central differences, full momentum decay | 19,956 |
| JAGUAR Muon (SVD) | One coordinate refresh; other momentum entries retained | 19,999 |
| Subspace ZO-Muon (SVD) | 32 Gaussian forward differences in a left subspace of dimension eight | 19,999 |
| Norm-matched ZO-SGD | Rank-one measurements; direction norm equals the compact polar norm of its own momentum | 19,956 |

The two cited controls implement the measurement/update rules of [Petrov et al., Section 3.3](https://arxiv.org/html/2506.04430v4) and [Lang et al., equations (10)–(11)](https://arxiv.org/html/2602.17155v2). Compact SVD replaces approximate orthogonalization, and the manuscript's common step schedule is used. The subspace basis is an orthonormal Gaussian QR basis. The subspace method has no momentum; its projection is resampled every 100 updates. These are stated adaptations for the CT comparison, not the authors' released neural-network benchmark implementations.

Tuning uses seeds 707 and 808. Evaluation uses seeds 901–905. The initial grid is `{0.001, 0.003, 0.01, 0.03, 0.1}`. After its lower boundary was selected in JAGUAR tuning, four smaller steps `{0.00001, 0.00003, 0.0001, 0.0003}` were added for that method before evaluation. Selection minimizes mean terminal objective on tuning seeds only. Selected steps are interior to the final grids and are stored in `results/selected_steps.json`.

The revision adds 96 tuning runs (1,918,184 calls), 40 evaluation runs (799,100 calls), 25 sensitivity runs (499,285 calls), and 14 serial timing runs (266,544 calls). Tuning, evaluation, sensitivity, and timing costs are reported separately. The common radius is `1e-3`; subspace Gaussian directions are unnormalized as in the cited estimator, whereas the original rank-one directions have unit Frobenius norm.

Sensitivity runs change one quantity at a time on the 20-view problem: probe counts 8, 32, 64 at radius `1e-3`, or radii `1e-4`, `1e-3`, `1e-2` at 32 probes. The initial step remains 0.01, with the common decay schedule, and is not retuned. Radius insensitivity is expected for exact central differences of this quadratic; it is not evidence about inexact forward solves.

## Scaling and accuracy checks

`numerical_diagnostics.py` checks coordinate recovery and V2 on a common iterate path and on independently advanced paths. It saves per-iteration scaling discrepancies, direction differences, singular-value ratios, threshold hits, and endpoint values. Shared-path directions agree to below `7e-13`; no zero threshold is reached. Independent fixed-step endpoints are sensitive to roundoff and differ from archived endpoints. Original manuscript table values are retained as historical records, not replaced by these diagnostic reruns.

`query_targets.csv` reads the saved regression histories and reports the first call count reaching each relative-gap target. Empty entries mean the target was not reached within the recorded run. Full-gradient loss counts exclude gradient work and are not equivalent function-only budgets. The four query-budget figures use only derivative-free trajectories.

`rank_variance.json` checks the fixed-gradient variance identity at equal Frobenius norm for ranks 1, 2, and 8. It uses exact directional inner products, not objective calls. It does not claim recovery of unknown singular subspaces.

## Checks performed

`test_baselines.py` checks preservation of the original CT trajectories, oracle budgets, JAGUAR's first update, zero-gradient handling, and norm matching. `audit_checks.py` independently evaluates saved terminal objectives and residuals, checks all budgets and tuning selections, compares serial reruns with evaluation records, and optionally repeats eight evaluation runs from scratch. It also checks the momentum-tracking inequality on a smooth quadratic. The numerical inequality check supplements the derivation; it is not a formal proof.

The original CartPole driver has not been recovered. No new CartPole runs or orthogonality-only CartPole ablations are claimed.
