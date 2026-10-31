# Derivative-Free Muon for Matrix Optimization with Rank-One Perturbations

Code and numerical records for Pengcheng Xie's *Derivative-Free Muon for Matrix Optimization with Rank-One Perturbations*. The algorithm estimates a matrix gradient using normalized rank-one probes, accumulates full momentum, and applies a polar update.

## Setup

Use Python 3.12 and the pinned dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 download_data.py
```

The download helper obtains the public walnut measurements and checks MD5 and SHA-256. An existing dataset can be installed with `python3 download_data.py --local-file /path/to/Data82.mat`.

Data attribution: K. Hämäläinen et al., [Tomographic X-ray data of a walnut, version 1.0.0](https://doi.org/10.5281/zenodo.1254206); [acquisition documentation](https://arxiv.org/abs/1502.04064).

## Reproduction coverage

The public material distinguishes rerunnable experiments from archived records:

| Paper result | Entry point | Coverage |
| --- | --- | --- |
| Figure 1 and Table 3: matched-accuracy calls | `make rank-one` | Reruns 120 RO-DFO-Muon trajectories and four coordinate references; recomputes first-hit statistics. |
| Table 4: original fixed-iteration regression | `python3 muon_zero_order_reproduction/run_experiments.py --output /tmp/muon-regression-replay` | Drivers and settings are supplied. Fixed-step SVD endpoints are sensitive to the numerical environment; archived endpoints and fresh diagnostics are distinguished. |
| Figure 2 and Table 5: gradient-corruption regression | `python3 muon_zero_order_reproduction/run_noisy_gradient_experiment.py` | Experiment and figure-generation driver supplied. |
| Figure 3 and Table 6: neural-network test | `python3 muon_zero_order_reproduction/run_neural_noisy_gradient_experiment.py` | Experiment and figure-generation driver supplied. |
| Figures 4 and 5, Tables 7 and 8: CartPole | Saved CSV trajectories and JSON summaries | Records supplied; the original simulator driver is unavailable. Figure 4 can be replotted by `rank_one_study/plot_results.py`. This release does not provide end-to-end CartPole reruns. |
| Figure 6 and Table 9: measured walnut CT | `python3 realdata_ct/run_walnut.py --workers 1`, then `python3 realdata_ct/report_walnut.py` | Driver, selected steps, tuning/evaluation records, references, and data download helper supplied. Serial timings use `--phase timing --workers 1`. |
| Table 10: additional CT controls | `make controls` | Separate tuning/evaluation seeds and sensitivity/timing records supplied. |
| Table 1 and mathematical bounds | `make check` | Oracle-count tests and six mathematical regression checks; numerical checks supplement the derivations. |
| Table 2: regression settings | `muon_zero_order_reproduction/config.json` | Fixed problem and method parameters. |

Run commands from the repository root after setup. `make report` processes saved records; it does not rerun every optimizer. The noisy-gradient and neural-network drivers write their own experiment directories. For fresh runs, use a separate checkout and move its relevant `results/` directories aside first; CT runners otherwise resume completed jobs. Keep the frozen release unchanged when comparing fresh results.

## Algorithm and experiments

- `rank_one_study/rank_one_muon.py`: scalar-objective RO-DFO-Muon implementation.
- `rank_one_study/`: 30-seed query-efficiency study on each of four regression problems.
- `muon_zero_order_reproduction/`: regression and neural-network drivers, parameters, and saved control-study records.
- `realdata_ct/`: measured-data reconstruction and adjoint-based CG reference.
- `revision_checks/`: sampled-coordinate, JAGUAR and subspace adaptations, norm-matched controls, and numerical diagnostics.
- `submission_checks/`: finite-difference, polar-deficit, scaling, descent, and reference-solution checks.

```sh
make check
make report
make rank-one
```

`make check` runs 13 unit tests, audits the saved CT and regression results, and performs six mathematical checks. `make report` regenerates figures and complete CT tables from the saved results, including the focused CT figure and all baseline values. `make rank-one` reruns the 120 random-probe trajectories and four coordinate references before regenerating plots. `make controls` repeats the added tuning, evaluation, sensitivity and timing stages; completed records are resumed. Experiment-specific READMEs give complete commands and settings.

The raw MAT file is downloaded separately and excluded from Git. Saved objective histories, terminal matrices, tuning selections and environment records support the reported comparisons. CT tuning and evaluation seeds are disjoint. Original and added timing environments are recorded separately. CartPole plots use saved trajectories; the original simulator driver is unavailable.
