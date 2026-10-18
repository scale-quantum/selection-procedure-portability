"""PBO indicator and rank logits under a frequency-neutral performance measure.

Two questions, both about the "below median" indicator of the probability of
backtest overfitting (PBO; Bailey, Borwein, Lopez de Prado and Zhu):

A. On the six-piece chains of spp_ribbon_study.py (seeds 4000-4029), the
   indicator was computed on the raw sum. Does it change when the performance
   measure is frequency-neutral? Three measures, the same measure in-sample
   and out-of-sample:
     sum      total of the +1 / -1 outcomes (as in the paper's 4.6),
     avg      sum / trades (average trade),
     tstat    sum / sqrt(trades): outcomes are +-1, so the sum's standard
              error grows like sqrt(trades); this is a risk-comparable score.
   The full distribution of the rank logits is reported, not only the share
   at or below zero.

B. A controlled test. Selection skill is held fixed (the ordering coefficient
   of the test piece is 0.8, or 0.0 for the no-skill arm). The period level
   (shift) and the frequency spread across candidates are varied, and the
   share of replicates in which the IS champion lands below the OOS median is
   read for each measure. With frequency spread 0 every candidate trades the
   same number of times, so the sum and the average trade rank candidates
   identically and a frequency channel cannot exist.

Ties in the out-of-sample measure use the average rank (the standard
convention); the library's ``places`` breaks ties by row number.

Run from the repository root:
    python scripts/spp_neutral_metric_study.py
    python scripts/spp_neutral_metric_study.py --quick
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spp_catalog_six as c  # noqa: E402
from spp.measure import places, top_index  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spp_neutral_metric_study"

METRICS = ("sum", "avg", "tstat")
SEED0_CHAIN = 4000
N_CHAINS = 30
SEED0_CTRL = 60000
N_CTRL = 200
WORKERS = 8
MEAN_TRADES = 447.0

TRAIN_SPEC = (1.00, c.SHIFT_STRONG)
TEST_ALPHA = (0.80, 0.00)
SHIFTS = (-1.15, -0.85, -0.45, 0.05, 0.55)
SPREADS = (0.0, 1.0, 2.0)


def measure_values(total: np.ndarray, trades: np.ndarray, pieces: int) -> dict[str, np.ndarray]:
    return {
        "sum": total,
        "avg": total / trades,
        "tstat": total / np.sqrt(trades * pieces),
    }


def avg_rank(values: np.ndarray, who: int) -> float:
    """Average rank of values[who] among values, 1 = worst."""
    v = values[who]
    below = float(np.sum(values < v))
    equal = float(np.sum(values == v))
    return below + (equal + 1.0) / 2.0


def omega_lambda(values: np.ndarray, who: int) -> tuple[float, float]:
    w = avg_rank(values, who) / (len(values) + 1.0)
    return w, float(np.log(w / (1.0 - w)))


def split_record(is_vals, oos_vals, oos_total) -> dict:
    n = len(oos_total)
    champ = int(top_index(is_vals, 1 / n)[0])
    w, lam = omega_lambda(oos_vals, champ)
    chosen = top_index(is_vals, 0.10)
    return {
        "omega": w,
        "lam": lam,
        "red": bool(oos_total[champ] <= 0),
        "slice_place": float(np.median(places(oos_vals)[chosen])),
    }


def chain_job(seed: int) -> dict:
    nets, _, trades, _, _ = c.build_chain(seed, c.CHAIN)
    rec = {"seed": seed, "loo": [], "triples": []}
    for hold in range(6):
        is_idx = [j for j in range(6) if j != hold]
        is_total = np.sum([nets[j] for j in is_idx], axis=0)
        oos_total = nets[hold]
        is_m = measure_values(is_total, trades, 5)
        oos_m = measure_values(oos_total, trades, 1)
        rec["loo"].append(
            {"hold": hold + 1, **{m: split_record(is_m[m], oos_m[m], oos_total) for m in METRICS}}
        )
    for test_idx in combinations(range(6), 3):
        is_idx = [j for j in range(6) if j not in test_idx]
        is_total = np.sum([nets[j] for j in is_idx], axis=0)
        oos_total = np.sum([nets[j] for j in test_idx], axis=0)
        is_m = measure_values(is_total, trades, 3)
        oos_m = measure_values(oos_total, trades, 3)
        rec["triples"].append(
            {"test": [j + 1 for j in test_idx], **{m: split_record(is_m[m], oos_m[m], oos_total) for m in METRICS}}
        )
    return rec


def build_mask_spread(rng: np.random.Generator, edge: np.ndarray, spread: float):
    """c.build_mask with the trade counts' deviation from the mean scaled by ``spread``.

    The random calls are in the same order as in c.build_mask, so spread = 1
    reproduces it exactly.
    """
    n = c.N_CLUSTERS * c.PER
    assign = np.repeat(np.arange(c.N_CLUSTERS), c.PER)
    exposure = np.clip(rng.normal(0.40, 0.16, size=c.N_CLUSTERS), 0.05, 0.85).astype(np.float32)
    residual = rng.normal(0, 1, size=(c.N_CLUSTERS, c.N_OPP)).astype(np.float32)
    width = np.sqrt(np.clip(1 - exposure ** 2, 0.02, None)).astype(np.float32)
    centers = exposure[:, None] * edge + width[:, None] * residual
    scores = centers[assign] + rng.normal(0, 0.22, size=(n, c.N_OPP)).astype(np.float32)
    base = rng.uniform(280, 640, size=c.N_CLUSTERS)
    stand = np.clip(base[assign] + rng.normal(0, 16, size=n), 180, 860).astype(np.int32)
    if spread == 1.0:
        n_trades = stand
    else:
        n_trades = np.clip(MEAN_TRADES + spread * (stand - MEAN_TRADES), 60, 1000).astype(np.int32)
    mask = c.take(scores, n_trades)
    return mask, n_trades.astype(np.float64), assign


def control_job(args: tuple[int, int]) -> list[dict]:
    seed, h_idx = args
    spread = SPREADS[h_idx]
    rng = np.random.default_rng(SEED0_CTRL + seed * 10 + h_idx)
    edge = rng.normal(0.0, 1.0, size=c.N_OPP).astype(np.float32)
    mask, _, _ = build_mask_spread(rng, edge, spread)
    trades = np.maximum(mask.sum(axis=1), 1).astype(np.float64)
    n = len(trades)
    train_lat = c.latent(TRAIN_SPEC[0], TRAIN_SPEC[1], edge, rng)
    train_net, _ = c.outcomes_from_latent(mask, train_lat, rng)
    is_m = measure_values(train_net, trades, 1)
    champs = {m: int(top_index(is_m[m], 1 / n)[0]) for m in METRICS}
    chosen = {m: top_index(is_m[m], 0.10) for m in METRICS}
    rows = []
    for alpha in TEST_ALPHA:
        for shift in SHIFTS:
            lat = c.latent(alpha, shift, edge, rng)
            net, _ = c.outcomes_from_latent(mask, lat, rng)
            oos_m = measure_values(net, trades, 1)
            row = {
                "seed": seed,
                "spread": spread,
                "alpha": alpha,
                "shift": shift,
                "set_mean": float(net.mean()),
                "set_black": float(np.mean(net > 0)),
                "trade_sd": float(trades.std()),
            }
            for m in METRICS:
                w, lam = omega_lambda(oos_m[m], champs[m])
                row[m] = {
                    "omega": w,
                    "lam": lam,
                    "slice_place": float(np.median(places(oos_m[m])[chosen[m]])),
                    "champ_red": bool(net[champs[m]] <= 0),
                }
            rows.append(row)
    return rows


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def chain_report(rows: list[dict], old: list[dict] | None) -> list[str]:
    n = len(rows)
    L = [
        "## A. Chains: the below-median indicator and the rank logits under three measures",
        "",
        f"{n} chains of six pieces (seeds {SEED0_CHAIN}-{SEED0_CHAIN + n - 1}), the same as the ribbon study. "
        "Splits: one piece as test and five as training (6 per chain), and three versus three (all 20 combinations). "
        "The same measure is used in-sample (to choose the champion) and out-of-sample (to rank it). "
        "The indicator is omega < 0.5, i.e. the logit is below zero.",
        "",
        "### A1. One piece on test (6 splits per chain)",
        "",
        "| Test piece | Type | Indicator, sum | Indicator, avg | Indicator, tstat | Champion in the red, sum | Slice place, sum | Slice place, avg | Slice place, tstat |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    types = ["strong", "strong", "weak, order kept", "strong, inverted", "no order, medium", "no order, strong"]
    for h in range(6):
        cells = [r["loo"][h] for r in rows]
        L.append(
            f"| {h + 1} | {types[h]} | "
            + " | ".join(f"{sum(x[m]['omega'] < 0.5 for x in cells)}/{n}" for m in METRICS)
            + f" | {sum(x['sum']['red'] for x in cells)}/{n} | "
            + " | ".join(f"{np.mean([x[m]['slice_place'] for x in cells]):.0f}" for m in METRICS)
            + " |"
        )
    if old is not None:
        chk = []
        for h in range(6):
            now = sum(r["loo"][h]["sum"]["omega"] < 0.5 for r in rows)
            was = sum(r["loo"][h]["champ_place"] < 50 for r in old)
            chk.append((now, was))
        L += [
            "",
            "Check against the ribbon study (raw sum, place < 50 by row-order ties): "
            + ", ".join(f"piece {h + 1}: {a} vs {b}" for h, (a, b) in enumerate(chk))
            + ". Differences come from tie handling only.",
        ]
    L += ["", "### A2. Three pieces on test (20 splits per chain)", ""]
    L += [
        "| Measure | Indicator: mean over chains | SD across chains | Min | Max | Logit mean | Logit p10 / p50 / p90 | Champion in the red |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    for m in METRICS:
        per = np.array([np.mean([x[m]["omega"] < 0.5 for x in r["triples"]]) for r in rows])
        lam = np.array([x[m]["lam"] for r in rows for x in r["triples"]])
        red = np.mean([x[m]["red"] for r in rows for x in r["triples"]])
        q = np.quantile(lam, [0.1, 0.5, 0.9])
        L.append(
            f"| {m} | {per.mean():.2f} | {per.std():.2f} | {per.min():.2f} | {per.max():.2f} | {lam.mean():+.2f} | "
            f"{q[0]:+.2f} / {q[1]:+.2f} / {q[2]:+.2f} | {red:.2f} |"
        )
    L += [
        "",
        "A3. One piece on test, pooled over pieces and chains (the same quantities):",
        "",
        "| Measure | Indicator | Logit mean | Logit p10 / p50 / p90 |",
        "|---|---:|---:|---|",
    ]
    for m in METRICS:
        lam = np.array([x[m]["lam"] for r in rows for x in r["loo"]])
        q = np.quantile(lam, [0.1, 0.5, 0.9])
        L.append(f"| {m} | {np.mean(lam < 0):.2f} | {lam.mean():+.2f} | {q[0]:+.2f} / {q[1]:+.2f} / {q[2]:+.2f} |")
    return L


def control_report(rows: list[dict]) -> list[str]:
    n_seed = len({r["seed"] for r in rows})
    L = [
        "## B. Controlled test: skill fixed, level and frequency spread varied",
        "",
        f"{n_seed} worlds per frequency spread (seeds {SEED0_CTRL}+). Training piece: ordering 1.0, shift "
        f"{TRAIN_SPEC[1]:+.2f}. Test piece: ordering {TEST_ALPHA[0]} (skill present) or {TEST_ALPHA[1]} (no skill), "
        "level shift as listed. Frequency spread scales the candidates' deviation from the mean trade count "
        f"({MEAN_TRADES:.0f}): 0 = every candidate trades the same number of times, 1 = the testbed, "
        "2 = twice the testbed spread. The cell is the share of worlds in which the champion chosen by the measure "
        "lands below the OOS median of that measure.",
        "",
    ]
    for alpha in TEST_ALPHA:
        L += [
            f"### B{1 + TEST_ALPHA.index(alpha)}. Test ordering {alpha} "
            + ("(skill present)" if alpha > 0 else "(no skill; a neutral measure should give about 50%)"),
            "",
            "| Frequency spread | Shift | Set mean sum | Set in the black | Indicator, sum | Indicator, avg | Indicator, tstat |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for h in SPREADS:
            for s in SHIFTS:
                cells = [r for r in rows if r["alpha"] == alpha and r["spread"] == h and r["shift"] == s]
                L.append(
                    f"| {h:g} | {s:+.2f} | {np.mean([x['set_mean'] for x in cells]):+.0f} | "
                    f"{pct(np.mean([x['set_black'] for x in cells]))} | "
                    + " | ".join(pct(np.mean([x[m]['omega'] < 0.5 for x in cells])) for m in METRICS)
                    + " |"
                )
        L.append("")
    L += [
        "### B3. Level sensitivity: range of the indicator across the five shifts",
        "",
        "| Test ordering | Frequency spread | Range, sum | Range, avg | Range, tstat |",
        "|---:|---:|---:|---:|---:|",
    ]
    for alpha in TEST_ALPHA:
        for h in SPREADS:
            rng_cells = []
            for m in METRICS:
                vals = [
                    np.mean([x[m]["omega"] < 0.5 for x in rows if x["alpha"] == alpha and x["spread"] == h and x["shift"] == s])
                    for s in SHIFTS
                ]
                rng_cells.append(f"{100 * (max(vals) - min(vals)):.0f} pts")
            L.append(f"| {alpha} | {h:g} | " + " | ".join(rng_cells) + " |")
    L += [
        "",
        f"With {n_seed} worlds a share has a standard error of about {100 * 0.5 / np.sqrt(n_seed):.0f} points at 50% "
        "and less near 0 or 100, so ranges under 15 points are within noise.",
        "",
        "### B4. Top-10% slice place under the three measures (mean over worlds)",
        "",
        "| Test ordering | Frequency spread | Shift | Slice place, sum | Slice place, avg | Slice place, tstat |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for alpha in TEST_ALPHA:
        for h in SPREADS:
            for s in SHIFTS:
                cells = [x for x in rows if x["alpha"] == alpha and x["spread"] == h and x["shift"] == s]
                L.append(
                    f"| {alpha} | {h:g} | {s:+.2f} | "
                    + " | ".join(f"{np.mean([x[m]['slice_place'] for x in cells]):.0f}" for m in METRICS)
                    + " |"
                )
    return L


def main() -> None:
    quick = "--quick" in sys.argv
    n_chains = 4 if quick else N_CHAINS
    n_ctrl = 6 if quick else N_CTRL
    OUT.mkdir(parents=True, exist_ok=True)

    # a check that spread = 1 reproduces the testbed mask
    r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
    e1 = r1.normal(0.0, 1.0, size=c.N_OPP).astype(np.float32)
    e2 = r2.normal(0.0, 1.0, size=c.N_OPP).astype(np.float32)
    m_ref, t_ref, _, _ = c.build_mask(r1, e1)
    m_new, t_new, _ = build_mask_spread(r2, e2, 1.0)
    assert np.array_equal(m_ref, m_new) and np.array_equal(t_ref, t_new), "spread = 1 must reproduce the testbed"

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        chains = list(pool.map(chain_job, [SEED0_CHAIN + i for i in range(n_chains)], chunksize=1))
    print(f"chains: {len(chains)} in {time.time() - t0:.0f} s", flush=True)

    old = None
    old_path = HERE / "spp_ribbon_study" / "ribbons.json"
    if old_path.exists() and not quick:
        old = json.loads(old_path.read_text(encoding="utf-8"))
        old = [r for r in old if r["seed"] < SEED0_CHAIN + n_chains]

    t0 = time.time()
    jobs = [(s, h) for s in range(n_ctrl) for h in range(len(SPREADS))]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        ctrl = [row for rows in pool.map(control_job, jobs, chunksize=1) for row in rows]
    print(f"controlled worlds: {len(jobs)} in {time.time() - t0:.0f} s", flush=True)

    (OUT / "chains.json").write_text(json.dumps(chains), encoding="utf-8")
    (OUT / "controlled.json").write_text(json.dumps(ctrl), encoding="utf-8")
    head = [
        "# PBO indicator and rank logits under a frequency-neutral measure",
        "",
        "Measures: `sum` (total of +1 / -1 outcomes), `avg` (sum / trades), `tstat` (sum / sqrt(trades)). "
        "The same measure is used in-sample and out-of-sample. Ties in the OOS measure use the average rank.",
        "",
    ]
    text = "\n".join(head + chain_report(chains, old) + [""] + control_report(ctrl)) + "\n"
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
