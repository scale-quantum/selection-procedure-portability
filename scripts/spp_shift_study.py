"""Sensitivity to the calibration shifts of a piece's level.

The SHIFT_* constants were fitted on seeds 41 and 100 by eye ("plus share near
one half"). Here each constant is shifted up and down with the others held
fixed, 30 worlds per point (seeds 20000-20029; type S offsets the seed by 5000),
and we check whether the main contrast holds: the sign of the champion's sum
against the verdict on the top-10 place.

Three verdicts are read: the "sum" rule with the ordinary yardstick, the "sum"
rule at equal trade frequency (the main reading), and the "average trade" rule
at equal frequency. Thresholds come from the null worlds of
scripts/spp_worlds_study and are not refitted.

    python scripts/spp_shift_study.py
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
from spp_worlds_study import analyze, write_csv  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spp_shift_study"
SEED0 = 20000
N = 30
SECOND_TYPE_OFFSET = 5000
WORKERS = 6

# (key, heading, train order coefficient, train shift, test coefficient, test shift,
#  which parameter moves, values, whether order is alive on test, current value, seed offset)
SWEEPS = [
    ("M", "Random transfer, medium test: test shift (SHIFT_FLAT_MEDIUM)", 1.00, c.SHIFT_STRONG, 0.00, None, "test",
     (-0.4, -0.2, 0.0, 0.2, 0.4), False, c.SHIFT_FLAT_MEDIUM, 0),
    ("S", "Random transfer, strong test: test shift (SHIFT_FLAT_STRONG)", 0.00, c.SHIFT_FLAT_MEDIUM, 0.00, None, "test",
     (0.30, 0.45, 0.60, 0.75, 0.90), False, c.SHIFT_FLAT_STRONG, SECOND_TYPE_OFFSET),
    ("W", "Weak after strong: test shift (SHIFT_WEAK)", 1.00, c.SHIFT_STRONG, 0.80, None, "test",
     (-0.45, -0.65, -0.85, -1.05, -1.25), True, c.SHIFT_WEAK, 0),
    ("I", "Inversion: test shift (SHIFT_STRONG_INVERTED)", 1.00, c.SHIFT_WEAK, -0.72, None, "test",
     (0.35, 0.55, 0.75, 0.95, 1.15), False, c.SHIFT_STRONG_INVERTED, 0),
    ("IT", "Inversion: train shift (SHIFT_WEAK)", 1.00, None, -0.72, c.SHIFT_STRONG_INVERTED, "train",
     (-0.45, -0.65, -0.85, -1.05, -1.25), False, c.SHIFT_WEAK, 0),
]


def tasks() -> list[tuple]:
    out = []
    for key, _t, a1, s1, a2, s2, which, values, _alive, _cur, off in SWEEPS:
        for v in values:
            for i in range(N):
                train_shift = v if which == "train" else s1
                test_shift = v if which == "test" else s2
                out.append((f"{key}|{v}", SEED0 + i + off, a1, train_shift, a2, test_shift))
    return out


def verdict(place, lo, hi):
    if place >= hi:
        return "перенос"
    if place <= lo:
        return "инверсия"
    return "нет вывода"


def main() -> None:
    thr = json.loads((HERE / "spp_worlds_study" / "thresholds.json").read_text(encoding="utf-8"))["thresholds"]
    views = (
        ("sum", "place_net", "sum|net|10", "\"sum\", ordinary"),
        ("sum", "placem_net", "sum|netm|10", "\"sum\", at equal frequency"),
        ("avg", "placem_net", "avg|netm|10", "\"average trade\", at equal frequency"),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(analyze, tasks(), chunksize=1))
    print(f"{len(rows)} worlds in {time.time() - t0:.0f} s", flush=True)
    write_csv(OUT / "worlds.csv", rows)
    L = [
        "# Sensitivity to calibration shifts",
        "",
        f"{N} worlds per point, seeds {SEED0}–{SEED0 + N - 1}. Top-10 place thresholds from the null worlds: "
        + "; ".join(f"{title} {thr[key][0]:.0f} / {thr[key][1]:.0f}" for _r, _f, key, title in views)
        + ". The value used in the article is marked with an asterisk.",
        "",
    ]
    for key, title, _a1, _s1, _a2, _s2, _which, values, alive, current, _off in SWEEPS:
        L += [
            f"## {title}",
            "",
            f"Order on test is {'alive' if alive else 'not alive'}: the correct decision is to {'keep' if alive else 'stop'}.",
            "",
            "| Value | Test family: mean / in the plus | Champion «+» | Champion place (ordinary) | Top-10 place (ordinary / at equal frequency) | "
            + " | ".join(f"{t}: transfer / no conclusion / inversion" for _r, _f, _k, t in views)
            + " | Correct: scalar / " + " / ".join(t for _r, _f, _k, t in views) + " |",
            "|---|---|---:|---:|---|" + "---|" * len(views) + "---|",
        ]
        for v in values:
            rs = [r for r in rows if r["tag"] == f"{key}|{v}"]
            n = len(rs)
            sign = np.array([r["sum_champ_pos"] for r in rs], dtype=int)
            cells = []
            oks = [int(np.sum(sign == int(alive)))]
            for rule, field, thr_key, _t in views:
                lo, hi = thr[thr_key]
                ver = [verdict(r[f"{rule}_f10_{field}"], lo, hi) for r in rs]
                keep = np.array([x == "перенос" for x in ver])
                oks.append(int(np.sum(keep == alive)))
                cells.append(f"{ver.count('перенос')} / {ver.count('нет вывода')} / {ver.count('инверсия')}")
            mark = "*" if abs(v - current) < 1e-9 else ""
            g_raw = np.mean([r["sum_f10_place_net"] for r in rs])
            g_m = np.mean([r["sum_f10_placem_net"] for r in rs])
            L.append(
                f"| {v:+.2f}{mark} | {np.mean([r['fam_mean'] for r in rs]):+.1f} / {np.mean([r['fam_pos'] for r in rs]):.2f} | "
                f"{sign.sum()}/{n} | {np.mean([r['sum_champ_place_net'] for r in rs]):.0f} | {g_raw:.0f} / {g_m:.0f} | "
                + " | ".join(cells)
                + " | " + " / ".join(str(o) for o in oks) + " |"
            )
        L.append("")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
