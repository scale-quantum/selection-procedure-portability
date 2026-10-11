"""Competing measures on the same worlds and with the same threshold protocol.

Answers the review question: does the top-slice percentile at equal frequency
give something beyond known measures of order. Worlds and seeds are the same as
in spp_worlds_study.py (null 2000-2199, held-out 8000-8199, configurations
1000-1029, the order curve 3000-3029). There are no new seeds, so the comparison
is on the same worlds. SPP columns are taken from spp_worlds_study/worlds.csv,
not recomputed.

Measures:
    sign      sign of the chosen name's sum (no threshold: keep on a plus);
    pbo       the chosen name is not below the middle on test (PBO indicator, no threshold);
    spp_place median of the top-10 percentiles at equal frequency (the article's main reading);
    spp_gap   top-10 gap at equal frequency, in set SD;
    ic        Spearman between the train score and the test sum on the whole set;
    ic_m      the same Spearman inside ten trade-count bins, averaged with bin-size weights;
    overlap   share of names in common between the train top 10% and the test top 10%;
    spread    test mean of the top train decile minus the bottom, in set SD.

For continuous measures the threshold is taken as for SPP: the 5% and 95% of the
null worlds (both types together). Transfer is at least the upper threshold;
inversion is at most the lower one. The second variant uses thresholds from
type M only. Output is spp_baseline_study/results.json and report.md.

    python scripts/spp_baseline_study.py
    python scripts/spp_baseline_study.py --quick
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
import spp_catalog_six as c  # noqa: E402
import spp_shift_study as ss  # noqa: E402
import spp_worlds_study as sw  # noqa: E402
from spp.measure import _slots, decile_table, spearman, top_index  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spp_baseline_study"
WORLDS_CSV = HERE / "spp_worlds_study" / "worlds.csv"
SHIFT_CSV = HERE / "spp_shift_study" / "worlds.csv"
WORKERS = 10
N_CAND = c.N_CLUSTERS * c.PER
K10 = int(round(0.10 * N_CAND))

# (key, caption, column in the joined table)
CONTINUOUS = (
    ("spp_place", "SPP: top-10 percentile at equal frequency", "spp_place"),
    ("spp_gap", "SPP: top-10 gap at equal frequency (SD)", "spp_gap"),
    ("ic", "Spearman train → test", "ic"),
    ("ic_m", "Spearman inside trade-count bins", "ic_m"),
    ("overlap", "Overlap of the top 10%", "overlap"),
    ("spread", "Top minus bottom decile (SD)", "spread"),
)
BINARY = (
    ("sign", "Sign of the chosen name «+»"),
    ("pbo", "Chosen name not below the middle (PBO indicator)"),
)
TAGS_CFG = [f"cfg{i}" for i in range(1, 6)]
SHIFT_TAGS = [f"{sweep[0]}|{v}" for sweep in ss.SWEEPS for v in sweep[7]]
TAGS_POW = [f"pow_{lvl}_{a:.1f}" for lvl in ("strong", "medium") for a in sw.POWER_ALPHAS]


def analyze(task: tuple) -> dict:
    tag, seed, a1, s1, a2, s2 = task
    w = c.build_pair(seed, a1, s1, a2, s2)
    tr, te, trades = w["train_net"], w["test_net"], w["trades"]
    slot, bins = _slots(te, trades, 10)
    rhos, weights = [], []
    for k in range(bins):
        members = np.flatnonzero(slot == k)
        r = spearman(tr[members], te[members]) if members.size > 2 else float("nan")
        if np.isfinite(r):
            rhos.append(r)
            weights.append(members.size)
    ic_m = float(np.average(rhos, weights=weights)) if rhos else float("nan")
    chosen_tr = set(top_index(tr, 0.10).tolist())
    chosen_te = set(top_index(te, 0.10).tolist())
    dec = decile_table(tr, te)
    sd = float(te.std())
    return {
        "tag": tag,
        "seed": seed,
        "ic": spearman(tr, te),
        "ic_m": ic_m,
        "overlap": len(chosen_tr & chosen_te) / K10,
        "spread": (dec[0]["mean"] - dec[-1]["mean"]) / sd if sd > 0 else float("nan"),
    }


def all_tasks() -> list[tuple]:
    return sw.null_tasks() + sw.hold_tasks() + sw.cfg_tasks() + sw.power_tasks() + ss.tasks()


def read_worlds() -> dict[tuple[str, int], dict]:
    out = {}
    for path in (WORLDS_CSV, SHIFT_CSV):
        with path.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                row = {}
                for k, v in r.items():
                    if k == "tag":
                        row[k] = v
                    else:
                        try:
                            row[k] = float(v)
                        except ValueError:
                            row[k] = v
                out[(row["tag"], int(row["seed"]))] = row
    return out


def join(rows: list[dict], worlds: dict) -> list[dict]:
    """Add the SPP and sign columns from worlds.csv to the new measures."""
    fam_k = K10
    out = []
    for r in rows:
        w = worlds[(r["tag"], r["seed"])]
        n = N_CAND
        fam_sd = w["sum_f10_sd_net"] * np.sqrt(fam_k / ((n - fam_k) / (n - 1)))
        out.append({
            **r,
            "spp_place": w["sum_f10_placem_net"],
            "spp_gap": w["sum_f10_gapm_net"] / fam_sd,
            "sign": int(w["sum_champ_pos"]),
            "pbo": int(w["sum_champ_place_net"] >= 50),
            "champ_place": w["sum_champ_place_net"],
        })
    return out


def of(rows: list[dict], tag: str) -> list[dict]:
    return [r for r in rows if r["tag"] == tag]


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def verdicts(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.where(values >= hi, "transfer", np.where(values <= lo, "inversion", "none"))


def counts(v: np.ndarray) -> list[int]:
    return [int(np.sum(v == "transfer")), int(np.sum(v == "none")), int(np.sum(v == "inversion")), int(len(v))]


def summarize(rows: list[dict]) -> dict:
    null_m, null_s = of(rows, "nullM"), of(rows, "nullS")
    hold_m, hold_s = of(rows, "holdM"), of(rows, "holdS")
    res: dict = {"n_null": len(null_m), "n_hold": len(hold_m), "thresholds": {}, "hold": {}, "cfg": {}, "pow": {}, "shift": {}}
    for key, _label, colname in CONTINUOUS:
        thr = {}
        for name, pool in (("mixed", null_m + null_s), ("M", null_m)):
            q = np.quantile(col(pool, colname), [0.05, 0.95])
            thr[name] = [float(q[0]), float(q[1])]
        res["thresholds"][key] = thr
        res["hold"][key] = {}
        for name in ("mixed", "M"):
            lo, hi = thr[name]
            res["hold"][key][name] = {
                "M": counts(verdicts(col(hold_m, colname), lo, hi)),
                "S": counts(verdicts(col(hold_s, colname), lo, hi)),
            }
        res["cfg"][key] = {}
        res["pow"][key] = {}
        for name in ("mixed", "M"):
            lo, hi = thr[name]
            res["cfg"][key][name] = {t: counts(verdicts(col(of(rows, t), colname), lo, hi)) for t in TAGS_CFG}
            res["pow"][key][name] = {t: counts(verdicts(col(of(rows, t), colname), lo, hi)) for t in TAGS_POW}
        lo, hi = thr["mixed"]
        res["shift"][key] = {t: counts(verdicts(col(of(rows, t), colname), lo, hi)) for t in SHIFT_TAGS}
    for key, _label in BINARY:
        res["hold"][key] = {"M": int(col(hold_m, key).sum()), "S": int(col(hold_s, key).sum()), "n": len(hold_m)}
        res["cfg"][key] = {t: [int(col(of(rows, t), key).sum()), len(of(rows, t))] for t in TAGS_CFG}
        res["pow"][key] = {t: [int(col(of(rows, t), key).sum()), len(of(rows, t))] for t in TAGS_POW}
        res["shift"][key] = {t: [int(col(of(rows, t), key).sum()), len(of(rows, t))] for t in SHIFT_TAGS}
    return res


def fmt_thr(v: list[float], key: str) -> str:
    d = 0 if key == "spp_place" else 2
    return f"{v[0]:.{d}f} / {v[1]:.{d}f}"


def build_report(res: dict) -> str:
    L = [
        "# Competing measures on the same worlds",
        "",
        f"Null worlds per type {res['n_null']}, held-out {res['n_hold']}; configurations and the order curve are {sw.N_WORLDS} worlds. "
        "Thresholds are the 5% / 95% of the null worlds (both types together, \"mixed\") or type M only.",
        "",
        "## Thresholds (lower / upper)",
        "",
        "| Measure | both types | type M |",
        "|---|---:|---:|",
    ]
    for key, label, _ in CONTINUOUS:
        t = res["thresholds"][key]
        L.append(f"| {label} | {fmt_thr(t['mixed'], key)} | {fmt_thr(t['M'], key)} |")
    L += ["", "## Held-out null worlds: false transfer / false inversion", "",
          "| Measure | thresholds: both types, M | both types, S | type-M thresholds, M | type M, S |", "|---|---:|---:|---:|---:|"]
    for key, label, _ in CONTINUOUS:
        h = res["hold"][key]
        cells = []
        for name in ("mixed", "M"):
            for typ in ("M", "S"):
                x = h[name][typ]
                cells.append(f"{x[0]} / {x[2]} of {x[3]}")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    for key, label in BINARY:
        h = res["hold"][key]
        L.append(f"| {label} (share kept) | {h['M']} of {h['n']} | {h['S']} of {h['n']} | — | — |")
    L += ["", "## Transfer by configuration and along the order curve (thresholds: both types together)", "",
          "| Measure | config 1 | config 2 | config 3 | config 4 | config 5 | α = 0.2 (+0.05) | α = 0.2 (0.00) | α = 0.4 (0.00) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label, _ in CONTINUOUS:
        cf, pw = res["cfg"][key]["mixed"], res["pow"][key]["mixed"]
        L.append(f"| {label} | " + " | ".join(str(cf[t][0]) for t in TAGS_CFG) + " | "
                 + " | ".join(str(pw[t][0]) for t in ("pow_strong_0.2", "pow_medium_0.2", "pow_medium_0.4")) + " |")
    for key, label in BINARY:
        cf, pw = res["cfg"][key], res["pow"][key]
        L.append(f"| {label} | " + " | ".join(str(cf[t][0]) for t in TAGS_CFG) + " | "
                 + " | ".join(str(pw[t][0]) for t in ("pow_strong_0.2", "pow_medium_0.2", "pow_medium_0.4")) + " |")
    L += ["", "## Level-shift sweeps: transfer out of 30 (thresholds: both types together)", ""]
    for sweep_key, title in (("W", "Weak after strong (order kept)"), ("I", "Inversion"), ("M", "No order, medium test")):
        tags = [t for t in SHIFT_TAGS if t.startswith(sweep_key + "|")]
        L += [f"### {title}", "", "| Measure | " + " | ".join(t.split("|")[1] for t in tags) + " |", "|---|" + "---:|" * len(tags)]
        for key, label, _ in CONTINUOUS:
            L.append(f"| {label} | " + " | ".join(str(res["shift"][key][t][0]) for t in tags) + " |")
        for key, label in BINARY:
            L.append(f"| {label} | " + " | ".join(str(res["shift"][key][t][0]) for t in tags) + " |")
        L.append("")
    return "\n".join(L) + "\n"


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def main() -> None:
    quick = "--quick" in sys.argv
    if quick:
        sw.N_WORLDS, sw.N_NULL, ss.N = 4, 6, 3
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = all_tasks()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        raw = list(pool.map(analyze, tasks, chunksize=1))
    print(f"{len(raw)} worlds in {time.time() - t0:.0f} s", flush=True)
    rows = join(raw, read_worlds())
    suffix = "-quick" if quick else ""
    write_csv(OUT / f"worlds{suffix}.csv", rows)
    res = summarize(rows)
    (OUT / f"results{suffix}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / f"report{suffix}.md").write_text(build_report(res), encoding="utf-8")
    print(OUT / f"report{suffix}.md")


if __name__ == "__main__":
    main()
