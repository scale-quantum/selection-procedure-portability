"""A null that keeps clusters: permutation of whole clusters (labels are known on the bench).

The candidate null (a random sample of candidates) does not keep the concentration of
the top slice, and a threshold fitted for K = 50 does not carry to another K (4.9).
Here the null is built from the world itself: test clusters (a candidate's sum vector
and its trade count together) are reassigned whole to other train clusters; position
inside the cluster is kept. The top 10% by training stays the same set; only whose
test results it receives changes. Verdict: transfer if the share of permutations
whose statistic is at least the observed one is at most 5%; inversion is the
symmetric case. Measures: the top-10 percentile at equal frequency and Spearman at
equal frequency (bins by the trade count of whoever owns the test result).

Cluster labels are unknown on live data, so this is a bench procedure, not one for
someone else's data: it shows that the issue is the null, not the statistic.

Generator modes: "cluster" (as in the article: trade count is set on the cluster) and
"candidate" (each candidate's trade count is independent; see spp_catalog_seven).
Seeds: null worlds M from 9000, S from 14000, order 0.4 from 9500.

    python scripts/spp_clusternull_study.py                     # cluster mode
    python scripts/spp_clusternull_study.py --mode candidate
    python scripts/spp_clusternull_study.py --from-saved        # rebuild the report from saved worlds
    python scripts/spp_clusternull_study.py --quick
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
from spp.measure import places_matched, spearman_matched, top_index  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spp_clusternull_study"
KS = (2, 5, 10, 25, 50, 125)
N_PERM = 100
N_NULL = 40
N_ORDER = 30
ALPHA = 0.4
ALPHA_LEVEL = 0.05
WORKERS = 10
MODE = "cluster"


def stats(train, test, trades, chosen) -> tuple[float, float]:
    return float(np.median(places_matched(test, trades)[chosen])), spearman_matched(train, test, trades)


def world(task: tuple) -> dict:
    kind, k, seed, mode, n_perm = task
    n = 2500
    per = n // k
    kw = {"n_clusters": k, "trade_mode": mode}
    if kind == "nullM":
        w = s7.build_pair(seed, 1.00, c.SHIFT_STRONG, 0.00, c.SHIFT_FLAT_MEDIUM, **kw)
    elif kind == "nullS":
        w = s7.build_pair(seed, 0.00, c.SHIFT_FLAT_MEDIUM, 0.00, c.SHIFT_FLAT_STRONG, **kw)
    else:
        w = s7.build_pair(seed, 1.00, c.SHIFT_STRONG, ALPHA, c.SHIFT_FLAT_MEDIUM, **kw)
    tr, te, trades, assign = w["train_net"], w["test_net"], w["trades"], w["assign"]
    chosen = top_index(tr, 0.10)
    obs = stats(tr, te, trades, chosen)
    rng = np.random.default_rng(seed + 777)
    pos_in = np.arange(n) % per
    perm_stats = np.empty((n_perm, 2))
    for b in range(n_perm):
        pi = rng.permutation(k)
        idx = pi[assign] * per + pos_in
        perm_stats[b] = stats(tr, te[idx], trades[idx], chosen)
    out = {"kind": kind, "k": k, "seed": seed, "obs_place": obs[0], "obs_ic_m": obs[1]}
    for j, name in enumerate(("place", "ic_m")):
        out[f"p_up_{name}"] = float((1 + np.sum(perm_stats[:, j] >= obs[j])) / (n_perm + 1))
        out[f"p_dn_{name}"] = float((1 + np.sum(perm_stats[:, j] <= obs[j])) / (n_perm + 1))
    return out


def summarize(rows: list[dict], n_perm: int, mode: str) -> tuple[dict, str]:
    summary: dict = {"n_perm": n_perm, "alpha": ALPHA_LEVEL, "mode": mode, "rows": {}}
    L = [f"# Null with cluster permutation (mode {mode}, {n_perm} permutations, level {ALPHA_LEVEL:.0%})", "",
         "| K | Null M: false transfer (percentile / Spearman) | Null S: false transfer | Nulls M+S: false inversion | Order 0.4: transfer |",
         "|---:|---|---|---|---|"]
    for k in KS:
        cell = {}
        for kind in ("nullM", "nullS", "order"):
            rs = [r for r in rows if r["kind"] == kind and r["k"] == k]
            cell[kind] = {
                "n": len(rs),
                "up_place": int(sum(r["p_up_place"] <= ALPHA_LEVEL for r in rs)),
                "up_ic_m": int(sum(r["p_up_ic_m"] <= ALPHA_LEVEL for r in rs)),
                "dn_place": int(sum(r["p_dn_place"] <= ALPHA_LEVEL for r in rs)),
                "dn_ic_m": int(sum(r["p_dn_ic_m"] <= ALPHA_LEVEL for r in rs)),
            }
        summary["rows"][str(k)] = cell
        m, s, o = cell["nullM"], cell["nullS"], cell["order"]
        nn = m["n"] + s["n"]
        L.append(
            f"| {k} | {m['up_place']}/{m['n']} ; {m['up_ic_m']}/{m['n']} | {s['up_place']}/{s['n']} ; {s['up_ic_m']}/{s['n']} | "
            f"{m['dn_place'] + s['dn_place']}/{nn} ; {m['dn_ic_m'] + s['dn_ic_m']}/{nn} | {o['up_place']}/{o['n']} ; {o['up_ic_m']}/{o['n']} |"
        )
    return summary, "\n".join(L) + "\n"


def main() -> None:
    global MODE
    quick = "--quick" in sys.argv
    if "--mode" in sys.argv:
        MODE = sys.argv[sys.argv.index("--mode") + 1]
    n_perm, n_null, n_order = N_PERM, N_NULL, N_ORDER
    if quick:
        n_perm, n_null, n_order = 30, 3, 3
    tag = "" if MODE == "cluster" else f"-{MODE}"
    suffix = tag + ("-quick" if quick else "")
    OUT.mkdir(parents=True, exist_ok=True)
    if "--from-saved" in sys.argv:
        rows = json.loads((OUT / f"worlds{suffix}.json").read_text(encoding="utf-8"))
    else:
        tasks = []
        for k in KS:
            tasks += [("nullM", k, 9000 + i, MODE, n_perm) for i in range(n_null)]
            tasks += [("nullS", k, 14000 + i, MODE, n_perm) for i in range(n_null)]
            tasks += [("order", k, 9500 + i, MODE, n_perm) for i in range(n_order)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            rows = list(pool.map(world, tasks, chunksize=1))
        print(f"{len(rows)} worlds in {time.time() - t0:.0f} s", flush=True)
        (OUT / f"worlds{suffix}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    summary, text = summarize(rows, n_perm, MODE)
    (OUT / f"results{suffix}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"report{suffix}.md").write_text(text, encoding="utf-8")
    print(OUT / f"report{suffix}.md")


if __name__ == "__main__":
    main()
