"""How many independent chances are in the set: cluster count at a fixed candidate count.

Candidates are always 2,500. Only the cluster split changes (K clusters of
2,500 / K similar candidates); everything else is the spp_catalog_six generator.
Two questions:

1. Null worlds (no order on test). How the width of the null distribution of
   the top-10 place and gap at equal frequency depends on K, and what false
   transfer rate the threshold fitted at K = 50 gives (spp_worlds_study).
2. Order is present on test (coefficient 0.4). How that same true link looks
   in the observed rank link across candidates and across cluster means, and
   how often transfer is seen.

Seeds: null worlds 9000-9099, order 9500-9529 (they do not overlap the others).

    python scripts/spp_cluster_study.py
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spp_catalog_six as c  # noqa: E402
from spp.measure import gap_matched, places_matched, spearman, top_index  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spp_cluster_study"
N_CAND = 2500
KS = (2, 5, 10, 25, 50, 125, 500)
SEED_NULL = 9000
N_NULL = 100
SEED_ORDER = 9500
N_ORDER = 30
ORDER_ALPHA = 0.4
WORKERS = 6


def world(task: tuple) -> dict:
    kind, k, seed = task
    c.N_CLUSTERS = k
    c.PER = N_CAND // k
    if kind == "null":
        w = c.build_pair(seed, 1.00, c.SHIFT_STRONG, 0.00, c.SHIFT_FLAT_MEDIUM)
    else:
        w = c.build_pair(seed, 1.00, c.SHIFT_STRONG, ORDER_ALPHA, c.SHIFT_FLAT_MEDIUM)
    tn, trades = w["test_net"], w["trades"]
    assign = w["assign"]
    chosen = top_index(w["train_net"], 0.10)
    pm = places_matched(tn, trades)
    ids, counts = np.unique(assign[chosen], return_counts=True)
    ta_mean = np.bincount(assign, weights=w["train_avg"]) / np.bincount(assign)
    te_mean = np.bincount(assign, weights=w["test_avg"]) / np.bincount(assign)
    return {
        "kind": kind,
        "k": k,
        "seed": seed,
        "placem10": float(np.median(pm[chosen])),
        "gapm10_sd": float(gap_matched(tn, chosen, trades)["gap_matched"] / tn.std()),
        "top10_clusters": int(len(ids)),
        "top10_eff": float(counts.sum() ** 2 / np.sum(counts ** 2)),
        "rho_cand": spearman(w["train_avg"], w["test_avg"]),
        "rho_cluster": spearman(ta_mean, te_mean) if k >= 5 else float("nan"),
        "champ_pos": int(tn[int(top_index(w["train_net"], 1 / len(tn))[0])] > 0),
    }


def main() -> None:
    thr = json.loads((HERE / "spp_worlds_study" / "thresholds.json").read_text(encoding="utf-8"))["thresholds"]
    lo, hi = thr["sum|netm|10"]
    tasks = [("null", k, SEED_NULL + i) for k in KS for i in range(N_NULL)]
    tasks += [("order", k, SEED_ORDER + i) for k in KS for i in range(N_ORDER)]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(world, tasks, chunksize=2))
    print(f"{len(rows)} worlds in {time.time() - t0:.0f} s", flush=True)
    (OUT / "worlds.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    L = [
        "# Cluster count at a fixed 2,500 candidates",
        "",
        f"Null worlds: {N_NULL} per K, seeds {SEED_NULL}–{SEED_NULL + N_NULL - 1}. "
        f"Order on test (coefficient {ORDER_ALPHA}): {N_ORDER} per K, seeds {SEED_ORDER}–{SEED_ORDER + N_ORDER - 1}. "
        f"Top-10 place threshold at equal frequency from the K = 50 null worlds of section 4.1: {lo:.0f} / {hi:.0f}.",
        "",
        "## Null worlds",
        "",
        "| K | candidates per cluster | clusters in the top 10% | effective clusters | gap SD (set SD) | place 5% / 50% / 95% | false transfer / inversion at the K = 50 threshold |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for k in KS:
        rs = [r for r in rows if r["kind"] == "null" and r["k"] == k]
        pl = np.array([r["placem10"] for r in rs])
        q5, q50, q95 = np.quantile(pl, [0.05, 0.5, 0.95])
        L.append(
            f"| {k} | {N_CAND // k} | {np.mean([r['top10_clusters'] for r in rs]):.1f} | {np.mean([r['top10_eff'] for r in rs]):.1f} | "
            f"{np.std([r['gapm10_sd'] for r in rs]):.2f} | {q5:.0f} / {q50:.0f} / {q95:.0f} | "
            f"{np.mean(pl >= hi):.2f} / {np.mean(pl <= lo):.2f} |"
        )
    L += [
        "",
        f"## Order on test (coefficient {ORDER_ALPHA})",
        "",
        "| K | average-trade rank link: by candidates | by cluster means | transfer at the K = 50 threshold | champion sign «+» |",
        "|---:|---:|---:|---:|---:|",
    ]
    for k in KS:
        rs = [r for r in rows if r["kind"] == "order" and r["k"] == k]
        pl = np.array([r["placem10"] for r in rs])
        cl = np.array([r["rho_cluster"] for r in rs])
        cl_txt = f"{np.nanmean(cl):+.2f}" if np.isfinite(cl).any() else "—"
        L.append(
            f"| {k} | {np.mean([r['rho_cand'] for r in rs]):+.2f} | {cl_txt} | "
            f"{int(np.sum(pl >= hi))}/{len(rs)} | {sum(r['champ_pos'] for r in rs)}/{len(rs)} |"
        )
    L.append("")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
