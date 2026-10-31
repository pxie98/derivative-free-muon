# Measured walnut CT experiment

This experiment reconstructs an attenuation image from physical X-ray measurements. It does not generate a phantom or simulate the observed sinogram. The original regression, neural-network, and CartPole records elsewhere in the project are separate studies.

Data: K. Hämäläinen, L. Harhanen, A. Kallonen, A. Kujanpää, E. Niemi, and S. Siltanen, *Tomographic X-ray data of a walnut*, Zenodo v1.0.0 (2015), https://doi.org/10.5281/zenodo.1254206. Documentation: https://arxiv.org/abs/1502.04064. The providers permit scientific use with attribution. The downloaded `data/Data82.mat` is unmodified, and its published MD5 is `5698942708300bc34b87931c6d91f6b6`.

Download URL, accessed 24 October 2026:
https://zenodo.org/records/1254206/files/Data82.mat?download=1

The file contains an 82 × 120 measured sinogram and a 9840 × 6724 sparse projection matrix. Vectorization is column-major. The 20-view case retains columns 0, 6, ..., 114; all remaining views are used only for reporting an out-of-fit residual. No extra noise, synthetic projections, positivity clipping, filtering, or ground-truth attenuation image is used.

The objective is `(||A vec(W) - b||² + 10 ||W||²)/(2 ||b||²)`. The fixed regularization value follows the documentation's example and is not optimized for image quality. The scalar objective performs one sparse forward application per query. Optimizers receive only this callable and the image dimension; only the separate CG reference and numerical audit use adjoint products. All polar factors are compact SVD factors with a relative rank threshold. No Newton–Schulz approximation is used.

Reproduction, from the project root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 realdata_ct/run_walnut.py --workers 4
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 realdata_ct/run_walnut.py --phase timing --workers 1
OPENBLAS_NUM_THREADS=1 python3 realdata_ct/audit_walnut.py
OPENBLAS_NUM_THREADS=1 python3 realdata_ct/report_walnut.py
```

Dependencies are pinned in `requirements.txt`; the actual runtime versions and platform are recorded in `environment.json` and `timing_environment.json`. For a completely fresh run, move `results/` aside first: the runner resumes completed jobs from their JSON and NPZ files. The tuning grid, seed lists, momentum, finite-difference radius, and budget are constants at the top of `run_walnut.py`.

There are 94 tuning runs: four randomized methods × five step sizes × two seeds × two view counts, plus seven step sizes × two view counts for deterministic coordinate recovery. Randomized steps are selected by mean terminal objective over seeds 707 and 808. Coordinate recovery tests two additional large steps because only one sweep fits in the budget. Evaluation uses five different seeds (901–905) per randomized method, plus one deterministic coordinate run per view count: 42 evaluation runs. The ten `timing_*` runs replay seed 901 serially; their iterates are checked against the corresponding evaluation runs. Times from concurrent tuning or evaluation jobs are not used in the manuscript. The reported machine is an Apple M3 Max, macOS 14.5, with one BLAS thread. Timing excludes the common dataset loading and reference diagnostics.

Each randomized run uses 19,956 objective/forward calls, including the initial objective and 307 updates of 65 calls each. Coordinate recovery uses 13,450 calls for one central-difference sweep plus initialization and endpoint evaluation. The remaining budget cannot fund a second full sweep. The momentum used by normalized ZO-SGD is driven by the same sequence of independent rank-one probes as rank-one Muon. Its direction has Frobenius norm sqrt(82), matching a full-rank polar direction; the first compact polar direction has lower rank.

Saved outputs include every run's terminal matrix and objective history (`*.npz`), full scalar summaries (`*.json`), selected step sizes, CG reference iterates, and selected step sizes. The reference is a regularized numerical solution, not a physical attenuation ground truth. The audit independently checks vectorization, finite-difference slopes against adjoint derivatives, call counts, all selected endpoints, held-out residuals, tuning/test separation, CG stationarity, and serial replays.

This is one specimen at one image resolution. Five optimizer seeds describe algorithmic variability, not variability across specimens. The results are a comparison for this fixed Tikhonov objective and do not establish superior diagnostic image quality, robustness across scanners, or large-scale performance. The adjoint-based CG reference solves this particular problem much more efficiently than the function-value methods.

## Figures

`report_walnut.py` generates `walnut_results.pdf` for the manuscript's comparison of RO-DFO-Muon and full coordinate recovery. The complete numerical table retains all five methods, including normalized ZO-SGD. Reconstructions use the same attenuation scale and the preassigned seed 901. Coordinate curves show the last completed update rather than interpolated intermediate recovery steps.
