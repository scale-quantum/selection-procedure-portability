"""Frequencies on fresh seeds: the sign of the chosen candidate against place and gap.

Built on spp_catalog_six.py (the same worlds, the same constants). It answers
questions that single seeds cannot show:

B1/B3  how many worlds out of 30, for each of the five configurations, disagree
       between the sign of the sum and the verdict on the group's place (2x2 tables, count of correct decisions);
B2     the same rule, two yardsticks: the "sum" rule and the "average trade" rule;
B4     the null distribution of gap and place, the transfer / no-conclusion /
       inversion threshold from the null worlds, sensitivity in the order coefficient;
B5     the "average trade" rule's gap in units of the average trade;
B11    gap and place at equal trade frequency (spp.gap_matched, spp.places_matched):
       the main reading; it does not depend on the piece's level.

Seeds do not overlap the calibration (41, 100-129): configurations 1000-1029,
null worlds 2000-2199, the sensitivity curve 3000-3029. The second world type
(medium training, strong test) offsets the seed by 5000, as in the catalog.

The gap is exact: the mean of the selected slice minus the mean of the whole family
(or minus the expectation of a random set with the same trade-frequency composition).
That is the expected mean of a random set drawn without replacement. Random sets
only supply the spread around it, so they are not drawn here. The check on 300
draws is the fields mc_gap / mc_sd.

    python scripts/spp_worlds_study.py
    python scripts/spp_worlds_study.py --quick
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
from spp.measure import (  # noqa: E402
    gap_against_random,
    gap_matched,
    places,
    places_matched,
    spearman,
    top_index,
)

OUT = Path(__file__).resolve().parent / "spp_worlds_study"
FRACS = (0.01, 0.05, 0.10, 0.30)
SEED_CFG = 1000
SEED_NULL = 2000
SEED_POWER = 3000
SEED_HOLD = 8000
SECOND_TYPE_OFFSET = 5000
POWER_ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8)
WORKERS = 6

N_WORLDS = 30
N_NULL = 200


def pct(f: float) -> int:
    return int(round(f * 100))


def analyze(task: tuple) -> dict:
    tag, seed, a1, s1, a2, s2 = task
    w = c.build_pair(seed, a1, s1, a2, s2)
    tn, ta = w["test_net"], w["test_avg"]
    trades = w["trades"]
    n = len(tn)
    pn, pa = places(tn), places(ta)
    pm = places_matched(tn, trades)
    mean_n, mean_a = float(tn.mean()), float(ta.mean())
    var_n, var_a = float(tn.var()), float(ta.var())
    rec: dict = {
        "tag": tag,
        "seed": seed,
        "fam_mean": mean_n,
        "fam_pos": float(np.mean(tn > 0)),
        "rho_net": spearman(w["train_net"], tn),
        "rho_avg": spearman(w["train_avg"], ta),
        "rho_trades": spearman(w["trades"], tn),
    }
    for rule, score in (("sum", w["train_net"]), ("avg", w["train_avg"])):
        champ = int(top_index(score, 1 / n)[0])
        rec[f"{rule}_champ_sum"] = float(tn[champ])
        rec[f"{rule}_champ_pos"] = int(tn[champ] > 0)
        rec[f"{rule}_champ_place_net"] = float(pn[champ])
        rec[f"{rule}_champ_place_avg"] = float(pa[champ])
        rec[f"{rule}_champ_placem_net"] = float(pm[champ])
        top5 = top_index(score, 5 / n)
        rec[f"{rule}_top5_mean"] = float(tn[top5].mean())
        rec[f"{rule}_top5_pos"] = int(tn[top5].mean() > 0)
        rec[f"{rule}_top5_place_net"] = float(np.median(pn[top5]))
        for f in FRACS:
            k = max(1, int(round(f * n)))
            chosen = top_index(score, f)
            key = f"{rule}_f{pct(f)}"
            fpc = (n - k) / (n - 1)
            sd_n = float(np.sqrt(var_n / k * fpc)) if k < n else float("nan")
            sd_a = float(np.sqrt(var_a / k * fpc)) if k < n else float("nan")
            sel_n, sel_a = tn[chosen], ta[chosen]
            gap_n = float(sel_n.mean()) - mean_n
            gap_a = float(sel_a.mean()) - mean_a
            rec[f"{key}_gap_net"] = gap_n
            rec[f"{key}_sd_net"] = sd_n
            rec[f"{key}_z_net"] = gap_n / sd_n
            rec[f"{key}_place_net"] = float(np.median(pn[chosen]))
            rec[f"{key}_pos"] = float(np.mean(sel_n > 0))
            rec[f"{key}_gap_avg"] = gap_a
            rec[f"{key}_sd_avg"] = sd_a
            rec[f"{key}_z_avg"] = gap_a / sd_a
            rec[f"{key}_place_avg"] = float(np.median(pa[chosen]))
            rec[f"{key}_gapm_net"] = gap_matched(tn, chosen, trades)["gap_matched"]
            rec[f"{key}_placem_net"] = float(np.median(pm[chosen]))
    chosen10 = top_index(w["train_net"], 0.10)
    _ids, _counts = np.unique(w["assign"][chosen10], return_counts=True)
    rec["top10_clusters"] = int(len(_ids))
    rec["top10_eff_clusters"] = float(_counts.sum() ** 2 / np.sum(_counts ** 2))
    mc = gap_against_random(tn, chosen10, draws=300, rng=np.random.default_rng(seed))
    rec["mc_gap"] = float(mc["gap"])
    rec["mc_sd"] = float(mc["random_sd"])
    return rec


def run(tasks: list[tuple], label: str) -> list[dict]:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(analyze, tasks, chunksize=1))
    print(f"{label}: {len(rows)} worlds in {time.time() - t0:.0f} s", flush=True)
    return rows


def null_tasks() -> list[tuple]:
    tasks = []
    for i in range(N_NULL):
        tasks.append(("nullM", SEED_NULL + i, 1.00, c.SHIFT_STRONG, 0.00, c.SHIFT_FLAT_MEDIUM))
        tasks.append(("nullS", SEED_NULL + i + SECOND_TYPE_OFFSET, 0.00, c.SHIFT_FLAT_MEDIUM, 0.00, c.SHIFT_FLAT_STRONG))
    return tasks


def hold_tasks() -> list[tuple]:
    """Held-out null worlds: thresholds were not fitted on them; the false-verdict rate is counted on them."""
    tasks = []
    for i in range(N_NULL):
        tasks.append(("holdM", SEED_HOLD + i, 1.00, c.SHIFT_STRONG, 0.00, c.SHIFT_FLAT_MEDIUM))
        tasks.append(("holdS", SEED_HOLD + i + SECOND_TYPE_OFFSET, 0.00, c.SHIFT_FLAT_MEDIUM, 0.00, c.SHIFT_FLAT_STRONG))
    return tasks


def cfg_tasks() -> list[tuple]:
    tasks = []
    for idx, (_name, _intended, a1, s1, a2, s2) in enumerate(c.CATALOG, start=1):
        shift = SECOND_TYPE_OFFSET if idx == 5 else 0
        for i in range(N_WORLDS):
            tasks.append((f"cfg{idx}", SEED_CFG + i + shift, a1, s1, a2, s2))
    return tasks


def power_tasks() -> list[tuple]:
    tasks = []
    for level, s2 in (("strong", c.SHIFT_STRONG), ("medium", c.SHIFT_FLAT_MEDIUM)):
        for a2 in POWER_ALPHAS:
            for i in range(N_WORLDS):
                tasks.append((f"pow_{level}_{a2:.1f}", SEED_POWER + i, 1.00, c.SHIFT_STRONG, a2, s2))
    return tasks


def col(rows: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in rows], dtype=float)


def of(rows: list[dict], tag: str) -> list[dict]:
    return [r for r in rows if r["tag"] == tag]


def verdict(place: float, lo: float, hi: float) -> str:
    if place >= hi:
        return "перенос"
    if place <= lo:
        return "инверсия"
    return "нет вывода"


# Yardsticks: (selection rule, what is measured, heading). Place field: place_<yard>; gap field: GAP[yard].
VIEWS = (
    ("sum", "net", "\"sum\", ordinary yardstick"),
    ("sum", "netm", "\"sum\", at equal frequency"),
    ("avg", "avg", "\"average trade\", average-trade yardstick"),
    ("avg", "netm", "\"average trade\", at equal frequency"),
)
PRIMARY = ("sum", "netm")
GAP_FIELD = {"net": "gap_net", "netm": "gapm_net", "avg": "gap_avg"}
PLACE_FIELD = {"net": "place_net", "netm": "placem_net", "avg": "place_avg"}


def place_key(rule: str, yard: str, fpc: int) -> str:
    return f"{rule}_f{fpc}_{PLACE_FIELD[yard]}"


def gap_key(rule: str, yard: str, fpc: int) -> str:
    return f"{rule}_f{fpc}_{GAP_FIELD[yard]}"


# Truth by configuration: whether order is alive on test (order coefficient on test > 0).
TRUTH_ALIVE = {1: True, 2: True, 3: False, 4: False, 5: False}
CFG_TITLES = {
    1: "1. Strong after strong, order kept",
    2: "2. Weak after strong, order kept",
    3: "3. Inversion (weak → strong, order flipped)",
    4: "4. Random transfer, medium test",
    5: "5. Random transfer, strong test",
}


def fmt_gap(v: float, yard: str, signed: bool = False) -> str:
    if yard == "avg":
        return f"{v:+.4f}" if signed else f"{v:.4f}"
    return f"{v:+.1f}" if signed else f"{v:.1f}"


def build_report(null_rows, cfg_rows, power_rows) -> tuple[str, dict]:
    L: list[str] = []
    thr: dict = {}
    both = null_rows
    L += [
        "# Frequencies on fresh seeds",
        "",
        f"Configurations: {N_WORLDS} worlds per configuration, seeds {SEED_CFG}–{SEED_CFG + N_WORLDS - 1} "
        f"(configuration 5 offset by {SECOND_TYPE_OFFSET}). Null worlds: {N_NULL} per type, "
        f"seeds {SEED_NULL}–{SEED_NULL + N_NULL - 1}. Sensitivity curve: {N_WORLDS} worlds per cell, "
        f"seeds {SEED_POWER}–{SEED_POWER + N_WORLDS - 1}.",
        "The gap is exact: the mean of the selected slice minus the mean of the whole family (ordinary yardstick) or minus the expectation of a random set "
        "with the same composition by trade-count deciles (at equal frequency). Place is the median place of the slice's members "
        "(at equal frequency, the place among candidates with a similar trade count). The main slice is the top 10%. "
        "The main reading is the \"sum\" rule, at equal frequency.",
        "",
        "## 1. Null worlds: what you get when there is no transfer",
        "",
        f"A null world has order coefficient 0 on test. Two types: M (strong training, medium test) and "
        f"S (medium training, strong test), {N_NULL} worlds each.",
        "",
    ]

    mc_diff = np.abs(col(both, "mc_gap") - col(both, "sum_f10_gap_net"))
    mc_sd = col(both, "mc_sd")
    an_sd = col(both, "sum_f10_sd_net")
    L += [
        f"Check of the exact gap against 300 random sets (\"sum\" rule, 10%, ordinary yardstick): "
        f"maximum |difference| {mc_diff.max():.2f}, median {np.median(mc_diff):.2f}; "
        f"SD of the random mean across draws {mc_sd.mean():.2f} against the analytic {an_sd.mean():.2f}.",
        "",
    ]

    for fpc in (1, 5, 10, 30):
        L += [f"### Slice {fpc}%", ""]
        L += [
            "| Yardstick | Type | gap: mean | gap: SD across worlds | place: 5% / 50% / 95% | SD of z (ordinary yardstick) | share |z|≥5 |",
            "|---|---|---:|---:|---|---:|---:|",
        ]
        for rule, yard, title in VIEWS:
            for name, rows in (("M", of(both, "nullM")), ("S", of(both, "nullS")), ("both", both)):
                gap = col(rows, gap_key(rule, yard, fpc))
                pl = col(rows, place_key(rule, yard, fpc))
                q5, q50, q95 = np.quantile(pl, [0.05, 0.50, 0.95])
                if yard in ("net", "avg"):
                    z = col(rows, f"{rule}_f{fpc}_z_{yard}")
                    z_cells = f"{z.std():.1f} | {np.mean(np.abs(z) >= 5):.2f}"
                else:
                    z_cells = "— | —"
                L.append(
                    f"| {title} | {name} | {fmt_gap(gap.mean(), yard)} | {fmt_gap(gap.std(), yard)} | "
                    f"{q5:.0f} / {q50:.0f} / {q95:.0f} | {z_cells} |"
                )
            pl = col(both, place_key(rule, yard, fpc))
            thr[f"{rule}|{yard}|{fpc}"] = (float(np.quantile(pl, 0.05)), float(np.quantile(pl, 0.95)))
        L.append("")

    L += [
        "Place thresholds (5% and 95% of the null distribution, both types together), top 10%:",
        "",
        "| Yardstick | lower threshold | upper threshold |",
        "|---|---:|---:|",
    ]
    for rule, yard, title in VIEWS:
        lo, hi = thr[f"{rule}|{yard}|10"]
        L.append(f"| {title} | {lo:.0f} | {hi:.0f} |")
    L += [
        "",
        "Verdict on the top-10 place: transfer if the place is at least the upper threshold; inversion if it is at most the lower one; otherwise no conclusion. "
        "Thresholds are fixed on the null worlds from 2000 and applied to other seeds.",
        "",
        "False-verdict rate on the null worlds (the same worlds that set the thresholds; each tail should be ≈5%) by type:",
        "",
        "| Yardstick | Type | false transfer | false inversion |",
        "|---|---|---:|---:|",
    ]
    for rule, yard, title in VIEWS:
        lo, hi = thr[f"{rule}|{yard}|10"]
        for name, rows in (("M", of(both, "nullM")), ("S", of(both, "nullS"))):
            pl = col(rows, place_key(rule, yard, 10))
            L.append(f"| {title} | {name} | {np.mean(pl >= hi):.3f} | {np.mean(pl <= lo):.3f} |")
    L += [
        "",
        "## 2. Configurations: sign of the sum against the verdict on place",
        "",
        "Scalar \"keep\" is the chosen candidate's sum on test above zero. Top-5 \"keep\" is the mean sum of the five best above zero. "
        "The SPP \"keep\" verdict is the top-10 place above the upper null threshold. "
        "The correct decision is to keep if order is alive on test (configurations 1 and 2), and to stop otherwise (3, 4, 5).",
        "",
    ]

    summary: dict = {}
    for idx in range(1, 6):
        rows = of(cfg_rows, f"cfg{idx}")
        n = len(rows)
        keep_truth = TRUTH_ALIVE[idx]
        L += [f"### {CFG_TITLES[idx]}", ""]
        fam = col(rows, "fam_mean")
        L.append(
            f"Test family: mean sum {fam.mean():+.1f}, share in the plus {col(rows, 'fam_pos').mean():.2f}. "
            f"Rank link of the average trade {col(rows, 'rho_avg').mean():+.2f}, of the sum {col(rows, 'rho_net').mean():+.2f}, "
            f"of trade count and sum {col(rows, 'rho_trades').mean():+.2f}."
        )
        L.append("")
        L += [
            "| Yardstick | Scalar: champion | Scalar: top 5 | SPP: transfer | SPP: no conclusion | SPP: inversion | Correct decisions: scalar / top 5 / SPP | False comfort: scalar / SPP | False alarm: scalar / SPP | Champion place | Top-10 place | Gap 10%: mean |",
            "|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|",
        ]
        cfg_summary = {}
        for rule, yard, title in VIEWS:
            lo, hi = thr[f"{rule}|{yard}|10"]
            pl_group = col(rows, place_key(rule, yard, 10))
            verdicts = [verdict(p, lo, hi) for p in pl_group]
            sign = col(rows, f"{rule}_champ_pos").astype(int)
            top5 = col(rows, f"{rule}_top5_pos").astype(int)
            spp_keep = np.array([v == "перенос" for v in verdicts])
            sc_ok = int(np.sum(sign == int(keep_truth)))
            t5_ok = int(np.sum(top5 == int(keep_truth)))
            sp_ok = int(np.sum(spp_keep == keep_truth))
            comfort_sc = int(np.sum((sign == 1) & (not keep_truth)))
            comfort_sp = int(np.sum(spp_keep & (not keep_truth)))
            alarm_sc = int(np.sum((sign == 0) & keep_truth))
            alarm_sp = int(np.sum((~spp_keep) & keep_truth))
            champ_field = {"net": "place_net", "avg": "place_avg", "netm": "placem_net"}[yard]
            champ_place = col(rows, f"{rule}_champ_{champ_field}").mean()
            gap = col(rows, gap_key(rule, yard, 10)).mean()
            L.append(
                f"| {title} | {sign.sum()}/{n} | {top5.sum()}/{n} | {verdicts.count('перенос')}/{n} | "
                f"{verdicts.count('нет вывода')}/{n} | {verdicts.count('инверсия')}/{n} | "
                f"{sc_ok} / {t5_ok} / {sp_ok} | {comfort_sc} / {comfort_sp} | {alarm_sc} / {alarm_sp} | "
                f"{champ_place:.0f} | {pl_group.mean():.0f} | {fmt_gap(gap, yard, True)} |"
            )
            cfg_summary[title] = {"scalar_ok": sc_ok, "top5_ok": t5_ok, "spp_ok": sp_ok, "n": n}
        summary[f"cfg{idx}"] = cfg_summary
        L.append("")

        rule, yard = PRIMARY
        lo, hi = thr[f"{rule}|{yard}|10"]
        sign = col(rows, "sum_champ_pos").astype(int)
        verdicts = [verdict(p, lo, hi) for p in col(rows, place_key(rule, yard, 10))]
        L += ["Main reading (\"sum\", at equal frequency): champion sign × verdict on the top-10 place.", "", "| | transfer | no conclusion | inversion |", "|---|---:|---:|---:|"]
        for label, s in (("sign «+»", 1), ("sign «−»", 0)):
            cells = [sum(1 for v, sg in zip(verdicts, sign) if sg == s and v == name) for name in ("перенос", "нет вывода", "инверсия")]
            L.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
        L.append("")

    L += [
        "## 3. Sensitivity: share of worlds with the verdict transfer by the order coefficient on test",
        "",
        "Training: order coefficient 1.00, shift +0.05. Thresholds are from the null worlds, section 1. "
        "The mean rank link of the average trade shows how the order coefficient turns into an observed link.",
        "",
        "| Test level | Order coefficient | Average-trade rank link | Champion by sum in the plus | \"sum\", ordinary: transfer | \"sum\", at equal frequency: transfer | \"average trade\", average-trade yardstick: transfer | \"average trade\", at equal frequency: transfer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in ("strong", "medium"):
        for a2 in POWER_ALPHAS:
            rows = of(power_rows, f"pow_{level}_{a2:.1f}")
            n = len(rows)
            cells = []
            for rule, yard, _t in VIEWS:
                lo, hi = thr[f"{rule}|{yard}|10"]
                cells.append(sum(p >= hi for p in col(rows, place_key(rule, yard, 10))))
            L.append(
                f"| {level} | {a2:.1f} | {col(rows, 'rho_avg').mean():+.2f} | {int(col(rows, 'sum_champ_pos').sum())}/{n} | "
                + " | ".join(f"{x}/{n}" for x in cells)
                + " |"
            )
    L.append("")
    return "\n".join(L) + "\n", {"thresholds": thr, "summary": summary}


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)


def main() -> None:
    global N_WORLDS, N_NULL
    quick = "--quick" in sys.argv
    if quick:
        N_WORLDS, N_NULL = 4, 6
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    null_rows = run(null_tasks(), "null worlds")
    cfg_rows = run(cfg_tasks(), "configurations")
    power_rows = run(power_tasks(), "sensitivity")
    hold_rows = run(hold_tasks(), "held-out null worlds")
    suffix = "-quick" if quick else ""
    write_csv(OUT / f"worlds{suffix}.csv", null_rows + cfg_rows + power_rows + hold_rows)
    text, extra = build_report(null_rows, cfg_rows, power_rows)
    (OUT / f"report{suffix}.md").write_text(text, encoding="utf-8")
    (OUT / f"thresholds{suffix}.json").write_text(json.dumps(extra, ensure_ascii=False, indent=1), encoding="utf-8")
    print(OUT / f"report{suffix}.md", f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
