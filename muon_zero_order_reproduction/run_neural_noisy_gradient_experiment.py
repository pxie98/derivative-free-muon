#!/usr/bin/env python3
"""Muon versus DFO on a single-hidden-layer neural-network matrix weight."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from muon_dfo import polar_factor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "neural_noisy_gradient"


class FixedReadoutNetwork:
    """Teacher--student tanh network; only the hidden matrix W is optimized."""

    def __init__(self, seed: int = 6606):
        rng = np.random.default_rng(seed)
        self.x = rng.normal(size=(6, 512))
        self.w_star = rng.normal(size=(8, 6)) / np.sqrt(6)
        self.readout = rng.normal(size=(4, 8)) / np.sqrt(8)
        self.y = self.readout @ np.tanh(self.w_star @ self.x)

    def objective(self, w: np.ndarray) -> float:
        residual = self.readout @ np.tanh(w @ self.x) - self.y
        return float(np.sum(residual * residual) / (2 * self.x.shape[1]))

    def gradient(self, w: np.ndarray) -> np.ndarray:
        hidden = np.tanh(w @ self.x)
        residual = self.readout @ hidden - self.y
        return ((self.readout.T @ residual) * (1 - hidden * hidden)) @ self.x.T / self.x.shape[1]


def run_muon(problem, w0, *, sigma, seed, iterations=300, lr=0.025, beta=0.9):
    rng = np.random.default_rng(seed)
    w, momentum = w0.copy(), np.zeros_like(w0)
    scale = np.linalg.norm(problem.gradient(w0)) / np.sqrt(w0.size)
    history = [problem.objective(w)]
    for _ in range(iterations):
        gradient = problem.gradient(w)
        gradient += sigma * scale * rng.normal(size=w.shape)
        momentum = beta * momentum + (1 - beta) * gradient
        w -= lr * polar_factor(momentum)
        history.append(problem.objective(w))
    return np.asarray(history)


def run_entrywise(problem, w0, *, iterations=300, lr=0.025, beta=0.9, h=1e-4):
    w, momentum = w0.copy(), np.zeros_like(w0)
    base, history, calls = problem.objective(w), [], 1
    history.append(base)
    for _ in range(iterations):
        gradient = np.empty_like(w)
        for index in np.ndindex(w.shape):
            perturbed = w.copy()
            perturbed[index] += h
            gradient[index] = (problem.objective(perturbed) - base) / h
            calls += 1
        momentum = beta * momentum + (1 - beta) * gradient
        w -= lr * polar_factor(momentum)
        base = problem.objective(w)
        calls += 1
        history.append(base)
    return np.asarray(history), calls


def run_low_rank(problem, w0, *, seed, iterations=300, lr=0.025, beta=0.9, h=1e-4, probes=6):
    rng = np.random.default_rng(seed)
    w, momentum = w0.copy(), np.zeros_like(w0)
    m, n = w.shape
    base, history, calls = problem.objective(w), [], 1
    history.append(base)
    for _ in range(iterations):
        surrogate = np.zeros_like(w)
        for _ in range(probes):
            u, v = rng.normal(size=m), rng.normal(size=n)
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
            direction = np.outer(u, v)
            surrogate += (problem.objective(w + h * direction) - base) / h * direction
            calls += 1
        surrogate *= (m * n) / probes
        momentum = beta * momentum + (1 - beta) * surrogate
        w -= lr * polar_factor(momentum)
        base = problem.objective(w)
        calls += 1
        history.append(base)
    return np.asarray(history), calls


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    problem = FixedReadoutNetwork()
    w0 = np.zeros_like(problem.w_star)
    levels, repetitions = [0.0, 0.1, 0.3, 1.0, 3.0], 30
    muon = {s: [run_muon(problem, w0, sigma=s, seed=12000 + r)
                for r in range(repetitions)] for s in levels}
    low_rank_runs = [run_low_rank(problem, w0, seed=14000 + r)[0]
                     for r in range(repetitions)]
    _, low_rank_calls = run_low_rank(problem, w0, seed=14000)
    entry, entry_calls = run_entrywise(problem, w0)

    rows = []
    for sigma, runs in muon.items():
        finals = np.array([r[-1] for r in runs])
        rows.append(dict(method="Noisy-gradient Muon", noise_level=sigma,
                         median=float(np.median(finals)), q25=float(np.quantile(finals, .25)),
                         q75=float(np.quantile(finals, .75)), evaluations=301, repetitions=30))
    finals = np.array([r[-1] for r in low_rank_runs])
    rows.append(dict(method="Low-rank DFO-Muon", noise_level=None,
                     median=float(np.median(finals)), q25=float(np.quantile(finals, .25)),
                     q75=float(np.quantile(finals, .75)), evaluations=low_rank_calls, repetitions=30))
    rows.append(dict(method="Entrywise DFO-Muon", noise_level=None,
                     median=float(entry[-1]), q25=float(entry[-1]), q75=float(entry[-1]),
                     evaluations=entry_calls, repetitions=1))
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps({
        "architecture": "y=A tanh(Wx), fixed A, optimized W in R^(8x6)",
        "samples": 512, "iterations": 300, "learning_rate": .025, "beta": .9,
        "fd_radius": 1e-4, "low_rank_probes": 6, "results": rows}, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(301)
    for sigma in levels:
        curves = np.stack(muon[sigma]); med = np.median(curves, axis=0)
        q1, q3 = np.quantile(curves, [.25, .75], axis=0)
        axes[0].semilogy(x, med, label=rf"Muon $\sigma={sigma:g}$")
        axes[0].fill_between(x, q1, q3, alpha=.12)
    lr = np.stack(low_rank_runs)
    axes[0].semilogy(x, np.median(lr, axis=0), "k--", lw=2, label="Low-rank DFO")
    axes[0].semilogy(x, entry, color="tab:orange", ls=":", lw=2, label="Entrywise DFO")
    axes[0].set(title=r"Training hidden matrix $W$", xlabel="Iteration", ylabel="Network MSE")
    axes[0].grid(True, which="both", alpha=.35); axes[0].legend(fontsize=8)
    medians = [rows[i]["median"] for i in range(len(levels))]
    axes[1].loglog(levels[1:], medians[1:], "o-", label="Noisy-gradient Muon")
    axes[1].axhline(rows[-2]["median"], color="black", ls="--", label="Low-rank DFO")
    axes[1].axhline(entry[-1], color="tab:orange", ls=":", label="Entrywise DFO")
    axes[1].set(title="Neural-network noise crossover", xlabel=r"Gradient-noise level $\sigma$",
                ylabel="Median final MSE")
    axes[1].grid(True, which="both", alpha=.35); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "neural_noisy_gradient.png", dpi=180)
    fig.savefig(OUT / "neural_noisy_gradient.pdf"); plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
