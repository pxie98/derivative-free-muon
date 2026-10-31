#!/usr/bin/env python3
"""Compare function-value DFO with Muon under a noisy gradient oracle."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from muon_dfo import entrywise_dfo_muon, low_rank_dfo_muon, noisy_gradient_muon
from run_experiments import make_problem

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "noisy_gradient"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = {
        "id": 5, "m": 8, "n": 8, "samples": 256, "target_rank": 2,
        "noise_std": 0.01, "regularization": 1e-4, "learning_rate": 0.035,
        "seed": 5505,
    }
    iterations, beta, fd_radius, probes = 200, 0.9, 1e-4, 4
    levels = [0.0, 0.1, 0.3, 1.0, 3.0]
    repetitions = 30
    problem, w0 = make_problem(cfg)
    common = dict(iterations=iterations, learning_rate=cfg["learning_rate"], beta=beta)

    entry = entrywise_dfo_muon(problem, w0, fd_radius=fd_radius, **common)
    low_rank_runs = [
        low_rank_dfo_muon(
            problem, w0, fd_radius=fd_radius, probes=probes,
            seed=7000 + rep, **common,
        )
        for rep in range(repetitions)
    ]
    noisy = {
        level: [
            noisy_gradient_muon(
                problem, w0, noise_level=level, seed=9000 + rep, **common,
            )
            for rep in range(repetitions)
        ]
        for level in levels
    }

    rows = []
    for level, runs in noisy.items():
        finals = np.array([run.objective[-1] for run in runs])
        errors = np.array([run.recovery_error[-1] for run in runs])
        rows.append({
            "method": "Noisy-gradient Muon", "noise_level": level,
            "median_final_objective": float(np.median(finals)),
            "q25_final_objective": float(np.quantile(finals, 0.25)),
            "q75_final_objective": float(np.quantile(finals, 0.75)),
            "median_recovery_error": float(np.median(errors)),
            "function_evaluations": int(runs[0].function_evaluations[-1]),
            "repetitions": repetitions,
        })
    for name, runs in [("Low-rank DFO-Muon", low_rank_runs), ("Entrywise DFO-Muon", [entry])]:
        finals = np.array([run.objective[-1] for run in runs])
        errors = np.array([run.recovery_error[-1] for run in runs])
        rows.append({
            "method": name, "noise_level": None,
            "median_final_objective": float(np.median(finals)),
            "q25_final_objective": float(np.quantile(finals, 0.25)),
            "q75_final_objective": float(np.quantile(finals, 0.75)),
            "median_recovery_error": float(np.median(errors)),
            "function_evaluations": int(runs[0].function_evaluations[-1]),
            "repetitions": len(runs),
        })

    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "summary.json").write_text(
        json.dumps({"case": cfg, "iterations": iterations, "beta": beta,
                    "fd_radius": fd_radius, "probes": probes,
                    "gradient_noise_definition":
                    "iid N(0, (sigma*||G(W0)||_F/sqrt(m*n))^2) per entry per iteration",
                    "results": rows}, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for level in levels:
        curves = np.stack([run.objective for run in noisy[level]])
        median = np.median(curves, axis=0)
        q25, q75 = np.quantile(curves, [0.25, 0.75], axis=0)
        x = np.arange(iterations + 1)
        axes[0].semilogy(x, median, label=rf"Muon $\sigma={level:g}$")
        axes[0].fill_between(x, q25, q75, alpha=0.12)
    lr_curves = np.stack([run.objective for run in low_rank_runs])
    axes[0].semilogy(np.median(lr_curves, axis=0), "k--", linewidth=2,
                     label="Low-rank DFO")
    axes[0].semilogy(entry.objective, color="tab:orange", linestyle=":",
                     linewidth=2, label="Entrywise DFO")
    axes[0].set(xlabel="Iteration", ylabel="Objective value",
                title="Convergence under gradient noise")
    axes[0].grid(True, which="both", alpha=0.35)
    axes[0].legend(fontsize=8)

    muon_medians = [rows[i]["median_final_objective"] for i in range(len(levels))]
    axes[1].loglog(levels[1:], muon_medians[1:], "o-", label="Noisy-gradient Muon")
    axes[1].axhline(np.median([r.objective[-1] for r in low_rank_runs]),
                    color="black", linestyle="--", label="Low-rank DFO")
    axes[1].axhline(entry.objective[-1], color="tab:orange", linestyle=":",
                    label="Entrywise DFO")
    axes[1].set(xlabel=r"Gradient-noise level $\sigma$",
                ylabel="Median final objective", title="Noise crossover")
    axes[1].grid(True, which="both", alpha=0.35)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "noisy_gradient_comparison.png", dpi=180)
    fig.savefig(OUT / "noisy_gradient_comparison.pdf")
    plt.close(fig)

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
