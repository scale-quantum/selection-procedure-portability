"""Cluster count (section 4.9) with the trade count set on the candidate, not the cluster.

In spp_catalog_six the trade-count base is set on the cluster, so the ten trade-count
bins partly coincide with cluster groups. This is the same experiment as
spp_cluster_study (2,500 candidates, K clusters; null worlds M from 9000, order 0.4
from seed 9500), in two modes: "cluster" (as in the article; it should reproduce
those numbers) and "candidate" (each candidate's trade count is independent). Two
readings: the top-10 percentile at equal frequency and Spearman at equal frequency;
thresholds come from the base bench at K = 50 (as in 4.9), not refitted.

    python scripts/spp_cluster_decoupled.py
    python scripts/spp_cluster_decoupled.py --quick
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spp_catalog_seven as s7  # noqa: E402
import spp_catalog_six as c  # noqa: E402
from spp.measure import gap_matched, places_matched, spearman, spearman_matched, top_index  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spp_cluster_decoupled"
KS = (2, 5, 10, 25, 50, 125, 500)
SEED_NULL, N_NULL = 9000, 100
SEED_ORDER, N_ORDER = 9500, 30
ALPHA = 0.4
WORKERS = 10


def world(task: tuple) -> dict:
    mode, kind, k, seed = task
    a2 = 0.0 if kind == "null" else ALPHA
    w = s7.build_pair(seed, 1.00, c.SHIFT_STRONG, a2, c.SHIFT_FLAT_MEDIUM, trade_mode=mode, n_clusters=k)
    tr, te, trades = w["train_net"], w["test_net"], w["trades"]
    chosen = top_index(tr, 0.10)
    _ids, counts = np.unique(w["assign"][chosen], return_counts=True)
    return {
        "mode": mode, "kind": kind, "k": k, "seed": seed,
        "place": float(np.median(places_matched(te, trades)[chosen])),
        "gap_sd": float(gap_matched(te, chosen, trades)["gap_matched"] / te.std()),
        "eff": float(counts.sum() ** 2 / np.sum(counts ** 2)),
        "ic_m": spearman_matched(tr, te, trades),
        "ic": spearman(tr, te),
        "trade_cluster_share": float(np.var(np.bincount(w["assign"], weights=trades) / np.bincount(w["assign"])) / np.var(trades)),
    }


def main() -> None:
    quick = "--quick" in sys.argv
    n_null, n_order = (4, 3) if quick else (N_NULL, N_ORDER)
    base = json.loads((HERE / "spp_baseline_study" / "results.json").read_text(encoding="utf-8"))["thresholds"]
    thr = {"place": base["spp_place"]["mixed"], "ic_m": base["ic_m"]["mixed"]}
    tasks = []
    for mode in ("cluster", "candidate"):
        for k in KS:
            tasks += [(mode, "null", k, SEED_NULL + i) for i in range(n_null)]
            tasks += [(mode, "order", k, SEED_ORDER + i) for i in range(n_order)]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(world, tasks, chunksize=2))
    print(f"{len(rows)} worlds in {time.time() - t0:.0f} s", flush=True)
    suffix = "-quick" if quick else ""
    (OUT / f"worlds{suffix}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    summary: dict = {"thresholds": thr, "rows": {}}
    L = ["# Cluster count: trades on the cluster and trades on the candidate", "",
         f"Base-bench thresholds: percentile {thr['place'][0]:.0f} / {thr['place'][1]:.0f}, Spearman at equal frequency {thr['ic_m'][0]:.2f} / {thr['ic_m'][1]:.2f}.", "",
         "| Mode | K | share of trade-count variance between clusters | effective clusters | gap SD (set SD) | Spearman SD | false transfer percentile | false transfer Spearman | transfer at α = 0.4 percentile | transfer at α = 0.4 Spearman |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for mode in ("cluster", "candidate"):
        for k in KS:
            rn = [r for r in rows if r["mode"] == mode and r["k"] == k and r["kind"] == "null"]
            ro = [r for r in rows if r["mode"] == mode and r["k"] == k and r["kind"] == "order"]
            cell = {
                "share": float(np.mean([r["trade_cluster_share"] for r in rn])),
                "eff": float(np.mean([r["eff"] for r in rn])),
                "sd_gap": float(np.std([r["gap_sd"] for r in rn])),
                "sd_ic_m": float(np.std([r["ic_m"] for r in rn])),
                "fp_place": int(sum(r["place"] >= thr["place"][1] for r in rn)),
                "fp_ic_m": int(sum(r["ic_m"] >= thr["ic_m"][1] for r in rn)),
                "fi_place": int(sum(r["place"] <= thr["place"][0] for r in rn)),
                "fi_ic_m": int(sum(r["ic_m"] <= thr["ic_m"][0] for r in rn)),
                "n_null": len(rn),
                "det_place": int(sum(r["place"] >= thr["place"][1] for r in ro)),
                "det_ic_m": int(sum(r["ic_m"] >= thr["ic_m"][1] for r in ro)),
                "n_order": len(ro),
            }
            summary["rows"][f"{mode}|{k}"] = cell
            L.append(
                f"| {mode} | {k} | {cell['share']:.2f} | {cell['eff']:.1f} | {cell['sd_gap']:.2f} | {cell['sd_ic_m']:.3f} | "
                f"{cell['fp_place']}/{cell['n_null']} | {cell['fp_ic_m']}/{cell['n_null']} | "
                f"{cell['det_place']}/{cell['n_order']} | {cell['det_ic_m']}/{cell['n_order']} |"
            )
    (OUT / f"results{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"report{suffix}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(OUT / f"report{suffix}.md")


if __name__ == "__main__":
    main()
