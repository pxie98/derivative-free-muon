#!/usr/bin/env python3
"""Run all four reconstructed matrix-regression experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from muon_dfo import (
    MatrixRegression,
    direct_structured_search,
    entrywise_dfo_muon,
    full_gradient_muon,
    low_rank_dfo_muon,
    low_rank_dfo_muon_v2,
)

ROOT = Path(__file__).resolve().parent


def make_problem(cfg: dict) -> tuple[MatrixRegression, np.ndarray]:
    rng = np.random.default_rng(int(cfg["seed"]))
    m, n, samples = int(cfg["m"]), int(cfg["n"]), int(cfg["samples"])
    rank = cfg.get("target_rank")
    if rank is None:
        w_star = rng.normal(size=(m, n)) / np.sqrt(n)
    else:
        left = rng.normal(size=(m, int(rank)))
        right = rng.normal(size=(int(rank), n))
        w_star = left @ right / np.sqrt(int(rank) * n)
    x = rng.normal(size=(n, samples))
    noise = float(cfg["noise_std"]) * rng.normal(size=(m, samples))
    y = w_star @ x + noise
    problem = MatrixRegression(x=x, y=y, w_star=w_star, regularization=float(cfg["regularization"]))
    return problem, np.zeros((m, n), dtype=float)


def run_case(global_cfg: dict, cfg: dict):
    problem, w0 = make_problem(cfg)
    common = dict(
        iterations=int(global_cfg["iterations"]),
        learning_rate=float(cfg["learning_rate"]),
        beta=float(global_cfg["beta"]),
        polar_method=str(global_cfg["polar_method"]),
    )
    return problem, [
        full_gradient_muon(problem, w0, **common),
        entrywise_dfo_muon(problem, w0, fd_radius=float(global_cfg["fd_radius"]), **common),
        low_rank_dfo_muon(
            problem, w0, fd_radius=float(global_cfg["fd_radius"]),
            probes=int(cfg["low_rank_probes"]), seed=int(cfg["seed"]) + 17, **common,
        ),
        low_rank_dfo_muon_v2(problem, w0, fd_radius=float(global_cfg["fd_radius"]), **common),
        direct_structured_search(
            problem, w0, iterations=int(global_cfg["iterations"]),
            learning_rate=float(cfg["direct_learning_rate"]),
            fd_radius=float(global_cfg["fd_radius"]),
            candidates=int(cfg["direct_candidates"]), seed=int(cfg["seed"]) + 31,
        ),
    ]


def save_case(out: Path, cfg: dict, problem: MatrixRegression, results: list) -> None:
    case_id = int(cfg["id"])
    titles = {
        1: r"Case 1: dense $W_\star$, low noise",
        2: r"Case 2: dense $W_\star$, higher noise",
        3: r"Case 3: low-rank $W_\star$, low noise",
        4: "Case 4: larger matrix",
    }
    styles = [
        dict(color="tab:blue", marker="o", linestyle="-"),
        dict(color="tab:orange", marker="s", linestyle="--"),
        dict(color="tab:green", marker="^", linestyle="-."),
        dict(color="tab:red", marker="*", linestyle=":"),
        dict(color="tab:pink", marker="D", linestyle="--"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for result, style in zip(results, styles):
        x = np.arange(result.objective.size)
        ax.semilogy(x, result.objective, label=result.method, markevery=20, linewidth=2, **style)
    ax.set(title=titles[case_id], xlabel="Iteration", ylabel="Objective value")
    ax.grid(True, which="both", alpha=0.45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"matrix_regression_case_{case_id}.png", dpi=180)
    fig.savefig(out / f"matrix_regression_case_{case_id}.pdf")
    plt.close(fig)

    with (out / f"matrix_regression_case_{case_id}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "iteration", "objective", "relative_objective", "recovery_error", "function_evaluations"])
        for result in results:
            for i in range(result.objective.size):
                writer.writerow([result.method, i, result.objective[i], result.relative_objective[i], result.recovery_error[i], result.function_evaluations[i]])

    summary = {
        "case": cfg,
        "ridge_optimum": problem.objective(problem.ridge_solution()),
        "methods": [
            {
                "method": r.method,
                "initial_objective": float(r.objective[0]),
                "final_objective": float(r.objective[-1]),
                "final_relative_objective": float(r.relative_objective[-1]),
                "final_recovery_error": float(r.recovery_error[-1]),
                "function_evaluations": int(r.function_evaluations[-1]),
            }
            for r in results
        ],
    }
    (out / f"matrix_regression_case_{case_id}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--case", type=int, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for case_cfg in cfg["cases"]:
        if args.case is not None and int(case_cfg["id"]) != args.case:
            continue
        problem, results = run_case(cfg, case_cfg)
        save_case(args.output, case_cfg, problem, results)
        print(f"completed case {case_cfg['id']}")


if __name__ == "__main__":
    main()
