# Mathematical submission checks

From the project root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 submission_checks/check_mathematics.py
```

The six checks cover finite-difference constants with bounded function error,
the polar optimality deficit of a truncated SVD, the corresponding weighted
descent estimate, the absolute zero cutoff, the scale-invariance assumption
needed for V2 equivalence, and the residual bound for the CG reference.
They include cases where applying the exact-polar statement to a thresholded
direction would fail, and an exact-maximizer selection that is not scale
invariant.

The CG check also reads the two saved measured-data reference reconstructions.
Obtain the verified `Data82.mat` with the project-level download helper first.
Results are written to `results/mathematics_audit.json`. These numerical
checks supplement the written derivations; they do not certify arbitrary
floating-point computations or establish a convergence rate for fixed
algorithm parameters. No original experiment records are overwritten.
