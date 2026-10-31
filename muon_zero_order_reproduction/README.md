# Derivative-Free Structured Muon: original experiment records

Regression and neural-network drivers, configuration, and archived numerical
records for Rank-One Derivative-Free Muon. Parameters are specified in
`config.json`. CartPole results are supplied as saved trajectories and summaries.

## Implemented methods

1. Full-gradient Muon.
2. Entrywise derivative-free Muon using coordinate forward differences.
3. Low-rank derivative-free Muon using normalized random rank-one probes.
4. Exhaustive basis-aligned rank-one probing (V2), a scaled coordinate method.
5. One-sided direct rank-one structured search.

`muon_dfo/core.py` follows the active pseudocode in the manuscript.  Every
black-box objective call is counted.  Both an SVD polar factor (stable reference
and default) and the short quintic Newton--Schulz iteration are implemented.

## Reproducibility choices

All settings are visible in `config.json`. The study fixes
independent random seeds and records:

- raw objective value (the quantity shown in the manuscript figures),
- relative objective gap (defined in the manuscript),
- relative recovery error (defined in the manuscript), and
- cumulative function-evaluation count.

The four cases preserve the documented distinctions:

| Case | Target | Noise | Dimension choice |
|---|---|---:|---:|
| 1 | dense | low | 8 x 8 |
| 2 | dense | higher | 8 x 8 |
| 3 | rank 2 | low | 8 x 8 |
| 4 | dense | low | 16 x 16 |

These dimensions are reported in the current manuscript.

## Run

```bash
cd muon_zero_order_reproduction
python3 -m pip install -r requirements.txt
python3 run_experiments.py
python3 run_noisy_gradient_experiment.py
python3 run_neural_noisy_gradient_experiment.py
python3 -m pytest -q
```

Run one case or choose another output directory:

```bash
python3 run_experiments.py --case 3
python3 run_experiments.py --config config.json --output results_alt
```

Outputs include PNG/PDF plots, complete CSV trajectories, and JSON summaries in
`results/`. Use another output directory to preserve the supplied records.

## Interpretation

The supplied records show the following behavior:

- entrywise DFO-Muon closely follows full-gradient Muon but uses many more
  function evaluations;
- rank-one probing gives useful descent at substantially lower query cost but
  plateaus above the entrywise methods;
- the rank-2 target case has a smaller sampled-method gap, but does not
  isolate target rank from other problem differences; and
- unguided direct rank-one search is the weakest baseline.

The independently evolved fixed-step polar iterations are sensitive to
roundoff. Fresh endpoint values can differ across numerical environments;
the archived CSV files supply the current manuscript's recorded values.
`../revision_checks/` checks the V2 scaling identity on shared paths and
reports fresh trajectories separately. CartPole summaries and trajectories
are available, but its original driver is unavailable.

## Noisy-gradient oracle experiment

`run_noisy_gradient_experiment.py` keeps function values exact but supplies
Muon with an additive gradient-noise floor.  It compares five noise levels over
30 seeds against entrywise and four-probe low-rank DFO.  The generated CSV,
JSON, PNG, and PDF files are stored in `results/noisy_gradient/`.

`run_neural_noisy_gradient_experiment.py` repeats the comparison on a
teacher--student neural network, optimizing only its `8 x 6` hidden weight
matrix while the readout is fixed.  Outputs are written to
`results/neural_noisy_gradient/`.
