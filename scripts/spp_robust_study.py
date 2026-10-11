"""What remains of the result once what is built into the generator is removed (spp_catalog_seven).

Generator variants:
    base           as in the article (check: worlds match spp_catalog_six);
    channel 0.5    a second level channel, shift-sensitivity spread 0.5;
    channel 1.0    the same, spread 1.0 (sensitivity from 0 to 2);
    trades ±30%    test trade count = train count * exp(0.3 * noise);
    channel + trades  both together;
    trades not on the cluster   each candidate's trade count is independent.

For each variant: 100 null worlds per type (seeds from 2000, type S offset by 5000, as in
spp_worlds_study), five configurations of 30 worlds (seeds from 1000, configuration 5 offset by 5000).
Matching is always by the train trade count, as on live windows. Measures: the top-10
percentile at equal frequency (SPP), Spearman at equal frequency, Spearman on the whole
set, and the sign of the chosen name. The verdict is read with two threshold sets: the
base bench (as in the article) and the variant's own null worlds (calibration "in place").

    python scripts/spp_robust_study.py
    python scripts/spp_robust_study.py --quick
"""

from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spp_catalog_seven as s7  # noqa: E402
import spp_catalog_six as c  # noqa: E402
from spp.measure import places_matched, spearman, spearman_matched, top_index  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spp_robust_study"
WORKERS = 10
N_NULL = 100
N_CFG = 30
SEED_NULL = 2000
SEED_CFG = 1000
OFFSET_S = 5000

VARIANTS = (
    ("base", "base", {}),
    ("beta05", "channel 0.5", {"beta_spread": 0.5}),
    ("beta10", "channel 1.0", {"beta_spread": 1.0}),
    ("trd30", "trades ±30%", {"test_trade_sigma": 0.3}),
    ("both", "channel 1.0 + trades ±30%", {"beta_spread": 1.0, "test_trade_sigma": 0.3}),
    ("cand", "trades not on the cluster", {"trade_mode": "candidate"}),
)
MEASURES = ("spp_place", "ic_m", "ic")


def tasks() -> list[tuple]:
    out = []
    for vkey, _vt, kw in VARIANTS:
        for i in range(N_NULL):
            out.append((vkey, "nullM", SEED_NULL + i, 1.00, c.SHIFT_STRONG, 0.00, c.SHIFT_FLAT_MEDIUM, kw))
            out.append((vkey, "nullS", SEED_NULL + i + OFFSET_S, 0.00, c.SHIFT_FLAT_MEDIUM, 0.00, c.SHIFT_FLAT_STRONG, kw))
        for idx, (_n, _i, a1, s1, a2, s2) in enumerate(c.CATALOG, start=1):
            off = OFFSET_S if idx == 5 else 0
            for i in range(N_CFG):
                out.append((vkey, f"cfg{idx}", SEED_CFG + i + off, a1, s1, a2, s2, kw))
    return out


def analyze(task: tuple) -> dict:
    vkey, tag, seed, a1, s1, a2, s2, kw = task
    w = s7.build_pair(seed, a1, s1, a2, s2, **kw)
    tr, te, trades = w["train_net"], w["test_net"], w["trades"]
    chosen = top_index(tr, 0.10)
    champ = int(top_index(tr, 1 / len(tr))[0])
    return {
        "variant": vkey,
        "tag": tag,
        "seed": seed,
        "spp_place": float(np.median(places_matched(te, trades)[chosen])),
        "ic_m": spearman_matched(tr, te, trades),
        "ic": spearman(tr, te),
        "sign": int(te[champ] > 0),
        "fam_pos": float(np.mean(te > 0)),
        "fam_mean": float(te.mean()),
    }


def verdict(v: float, lo: float, hi: float) -> str:
    return "transfer" if v >= hi else ("inversion" if v <= lo else "none")


def counts(vals, lo, hi) -> list[int]:
    v = [verdict(x, lo, hi) for x in vals]
    return [v.count("transfer"), v.count("none"), v.count("inversion"), len(v)]


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def summarize(rows: list[dict]) -> dict:
    base_thr = json.loads((HERE / "spp_baseline_study" / "results.json").read_text(encoding="utf-8"))["thresholds"]
    res: dict = {"base_thresholds": {m: base_thr[m]["mixed"] for m in MEASURES}, "variants": {}}
    for vkey, _vt, _kw in VARIANTS:
        vr = [r for r in rows if r["variant"] == vkey]
        nulls = [r for r in vr if r["tag"] in ("nullM", "nullS")]
        out: dict = {"own_thresholds": {}, "null_base": {}, "null_own": {}, "cfg_base": {}, "cfg_own": {}, "sign": {}, "fam": {}}
        for m in MEASURES:
            q = np.quantile(col(nulls, m), [0.05, 0.95])
            out["own_thresholds"][m] = [float(q[0]), float(q[1])]
            for label, thr in (("base", res["base_thresholds"][m]), ("own", out["own_thresholds"][m])):
                lo, hi = thr
                out[f"null_{label}"][m] = {t: counts(col([r for r in vr if r["tag"] == t], m), lo, hi) for t in ("nullM", "nullS")}
                out[f"cfg_{label}"][m] = {f"cfg{i}": counts(col([r for r in vr if r["tag"] == f"cfg{i}"], m), lo, hi) for i in range(1, 6)}
        for i in range(1, 6):
            rs = [r for r in vr if r["tag"] == f"cfg{i}"]
            out["sign"][f"cfg{i}"] = [int(col(rs, "sign").sum()), len(rs)]
            out["fam"][f"cfg{i}"] = [float(col(rs, "fam_mean").mean()), float(col(rs, "fam_pos").mean())]
        res["variants"][vkey] = out
    return res


def report(res: dict) -> str:
    title = {v[0]: v[1] for v in VARIANTS}
    L = ["# What survives once what is built into the generator is removed", "",
         f"Null worlds {N_NULL} per type, configurations of {N_CFG} worlds; matching by the train trade count. "
         "transfer on the base-bench thresholds: " + "; ".join(f"{m} {t[0]:.2f} / {t[1]:.2f}" for m, t in res["base_thresholds"].items()) + ".", ""]
    for label, key in (("base-bench thresholds", "base"), ("the variant's own null-world thresholds", "own")):
        L += [f"## transfer out of 30 (configurations 1–5) and false transfer on null worlds out of {N_NULL}: {label}", "",
              "| Variant | Measure | Config 1 | Config 2 | Config 3 | Config 4 | Config 5 | Null M | Null S |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for vkey, vt, _kw in VARIANTS:
            v = res["variants"][vkey]
            for m in MEASURES:
                cf, nl = v[f"cfg_{key}"][m], v[f"null_{key}"][m]
                L.append(f"| {vt} | {m} | " + " | ".join(str(cf[f'cfg{i}'][0]) for i in range(1, 6)) + f" | {nl['nullM'][0]} | {nl['nullS'][0]} |")
            L.append(f"| {vt} | sign «+» | " + " | ".join(str(v['sign'][f'cfg{i}'][0]) for i in range(1, 6)) + " | — | — |")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    quick = "--quick" in sys.argv
    global N_NULL, N_CFG
    if quick:
        N_NULL, N_CFG = 3, 3
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(analyze, tasks(), chunksize=1))
    print(f"{len(rows)} worlds in {time.time() - t0:.0f} s", flush=True)
    suffix = "-quick" if quick else ""
    with (OUT / f"worlds{suffix}.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    res = summarize(rows)
    (OUT / f"results{suffix}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"report{suffix}.md").write_text(report(res), encoding="utf-8")
    print(OUT / f"report{suffix}.md")


if __name__ == "__main__":
    main()
