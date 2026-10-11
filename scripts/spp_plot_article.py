"""Article figures from the study runs. Numbers are not typed by hand.

Figure 1: share of replicates in which the reading matched the preset ordering,
against the test-window shift (section 4.3).
Figure 2: share of replicates with verdict "carries over", against the
ordering coefficient on the test window (section 4.4).

Reads scripts/spp_baseline_study/results.json. Writes PNG files to
figures/ at the repository root.

    python scripts/spp_plot_article.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
RES = json.loads((HERE / "spp_baseline_study" / "results.json").read_text(encoding="utf-8"))

SERIES = (
    ("sign", "sign of the selected", "#9b2226", "o", "-"),
    ("ic", "rank IC, unmatched", "#ca6702", "s", "--"),
    ("ic_m", "rank IC, trade-count-matched", "#0a9396", "^", "-"),
    ("spp_place", "top 10% percentile, matched", "#001219", "D", "-"),
)
SWEEPS = (
    ("W", "weak after strong\n(ordering preserved)", (-0.45, -0.65, -0.85, -1.05, -1.25), True, -0.85),
    ("I", "ordering inverted", (0.35, 0.55, 0.75, 0.95, 1.15), False, 0.75),
    ("M", "no ordering,\nmiddle test", (-0.4, -0.2, 0.0, 0.2, 0.4), False, 0.0),
)


def keep(stat: str, group: str, tag: str) -> tuple[int, int]:
    x = RES[group][stat]
    if stat in ("sign", "pbo"):
        return x[tag][0], x[tag][1]
    y = x[tag] if group == "shift" else x["mixed"][tag]
    return y[0], y[3]


def fig1() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
    for ax, (key, title, values, alive, current) in zip(axes, SWEEPS):
        for stat, label, color, marker, ls in SERIES:
            ys = []
            for v in values:
                k, n = keep(stat, "shift", f"{key}|{v}")
                ys.append((k if alive else n - k) / n)
            ax.plot(values, ys, color=color, marker=marker, ls=ls, lw=1.8, ms=5, label=label)
        ax.axvline(current, color="#999999", lw=0.8, ls=":")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("test-window shift (period level)", fontsize=9)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("share matching the preset ordering", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "spp_level_curves.png", dpi=160)
    plt.close(fig)


def fig2() -> None:
    alphas = (0.0, 0.2, 0.4, 0.6, 0.8)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for ax, (lvl, title) in zip(axes, (("strong", "test shift +0.05"), ("medium", "test shift 0.00"))):
        for stat, label, color, marker, ls in SERIES:
            ys = []
            for a in alphas:
                k, n = keep(stat, "pow", f"pow_{lvl}_{a:.1f}")
                ys.append(k / n)
            ax.plot(alphas, ys, color=color, marker=marker, ls=ls, lw=1.8, ms=5, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("ordering coefficient on the test window", fontsize=9)
        ax.set_ylim(-0.03, 1.05)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel('share with verdict "carries over"', fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(OUT / "spp_power_curves.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print(OUT)
