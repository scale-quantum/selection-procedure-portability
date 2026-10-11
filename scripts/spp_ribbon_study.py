"""Thirty ribbons of six pieces: rolling steps and combinatorial splits.

Built on spp_catalog_six.py. The catalog has one ribbon (seed 41), and the
"four pluses out of five" conclusion is a property of one draw. Here the same
ribbon is repeated on 30 fresh seeds (4000-4029), and each step and split is
read two ways:

  the sign of the chosen candidate's sum (what a rolling window sees; in the
  splits, the shares "below the middle" and "in the minus", what is tied to
  PBO and the probability of a loss);
  the top-10 place and the group's gap over a random set of the same size.

Thresholds for the verdict transfer / no conclusion / inversion come from the
null worlds of scripts/spp_worlds_study (thresholds.json) and are not fitted here.

Truth for a step: order is alive if the order coefficient on test is > 0
(steps 1→2 and 2→3); otherwise it is not (3→4 is inversion; 4→5 and 5→6 have no order).

    python scripts/spp_ribbon_study.py
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
from spp.measure import gap_matched, places, places_matched, top_index  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spp_ribbon_study"
SEED0 = 4000
N_RIBBONS = 30
WORKERS = 6
ALIVE = [True, True, False, False, False]
STEP_ALPHA = [0.82, 0.78, -0.72, 0.00, 0.00]
STEP_NAME = ["1→2", "2→3", "3→4", "4→5", "5→6"]
TEST_PIECE_TYPE = ["strong", "weak", "strong (inversion)", "medium", "strong"]


def group_stats(score, test_vals, pl_test):
    n = len(test_vals)
    chosen = top_index(score, 0.10)
    return {
        "place": float(np.median(pl_test[chosen])),
        "gap": float(test_vals[chosen].mean() - test_vals.mean()),
    }


def ribbon(seed: int) -> dict:
    nets, avgs, trades, assign, _ = c.build_chain(seed, c.CHAIN)
    n = len(nets[0])
    pl_net = [places(x) for x in nets]
    pl_avg = [places(x) for x in avgs]
    pl_m = [places_matched(x, trades) for x in nets]
    rec: dict = {"seed": seed, "steps": [], "loo": [], "triples": []}
    for i in range(5):
        step = {}
        for rule, score_tr in (("sum", nets[i]), ("avg", avgs[i])):
            champ = int(top_index(score_tr, 1 / n)[0])
            chosen = top_index(score_tr, 0.10)
            g_net = group_stats(score_tr, nets[i + 1], pl_net[i + 1])
            g_avg = group_stats(score_tr, avgs[i + 1], pl_avg[i + 1])
            step[rule] = {
                "champ_sum": float(nets[i + 1][champ]),
                "champ_pos": int(nets[i + 1][champ] > 0),
                "champ_place": float(pl_net[i + 1][champ]),
                "g_place_net": g_net["place"],
                "g_gap_net": g_net["gap"],
                "g_place_avg": g_avg["place"],
                "g_gap_avg": g_avg["gap"],
                "g_placem": float(np.median(pl_m[i + 1][chosen])),
                "g_gapm": gap_matched(nets[i + 1], chosen, trades)["gap_matched"],
            }
        step["family_mean"] = float(nets[i + 1].mean())
        rec["steps"].append(step)
    for hold in range(6):
        train_score = np.sum([nets[j] for j in range(6) if j != hold], axis=0)
        champ = int(top_index(train_score, 1 / n)[0])
        te = nets[hold]
        g = group_stats(train_score, te, pl_net[hold])
        chosen = top_index(train_score, 0.10)
        rec["loo"].append({
            "hold": hold + 1,
            "champ_sum": float(te[champ]),
            "champ_place": float(pl_net[hold][champ]),
            "g_place": g["place"],
            "g_gap": g["gap"],
            "g_placem": float(np.median(pl_m[hold][chosen])),
            "g_gapm": gap_matched(te, chosen, trades)["gap_matched"],
        })
    for test_idx in combinations(range(6), 3):
        train_idx = [j for j in range(6) if j not in test_idx]
        train_score = np.sum([nets[j] for j in train_idx], axis=0)
        summed = np.sum([nets[j] for j in test_idx], axis=0)
        pl = places(summed)
        champ = int(top_index(train_score, 1 / n)[0])
        g = group_stats(train_score, summed, pl)
        chosen = top_index(train_score, 0.10)
        pm3 = places_matched(summed, trades)
        rec["triples"].append({
            "test": [j + 1 for j in test_idx],
            "champ_sum": float(summed[champ]),
            "champ_place": float(pl[champ]),
            "g_place": g["place"],
            "g_gap": g["gap"],
            "g_placem": float(np.median(pm3[chosen])),
            "g_gapm": gap_matched(summed, chosen, trades)["gap_matched"],
        })
    return rec


def verdict(place, lo, hi):
    if place >= hi:
        return "перенос"
    if place <= lo:
        return "инверсия"
    return "нет вывода"


def report(rows: list[dict], thr: dict) -> str:
    lo_s, hi_s = thr["sum|netm|10"]
    lo_a, hi_a = thr["avg|netm|10"]
    n = len(rows)
    L = [
        "# Thirty ribbons of six pieces",
        "",
        f"Ribbons: {n}, seeds {SEED0}–{SEED0 + n - 1}. Top-10 place thresholds at equal trade frequency from the null worlds: "
        f"the \"sum\" rule {lo_s:.0f} / {hi_s:.0f}, the \"average trade\" rule {lo_a:.0f} / {hi_a:.0f}.",
        "",
        "## 1. Rolling steps (training is the previous piece)",
        "",
        "Scalar \"keep\" is the champion's sum on test > 0. SPP \"keep\" is the verdict transfer. "
        "The correct decision is to keep if order is alive on test.",
        "",
        "| Step | Test | Order coefficient | Order alive | Champion «+» | Champion place | Top-10 place (at equal frequency) | Top-10 gap (at equal frequency) | transfer / no conclusion / inversion (\"sum\" rule) | Correct: scalar / SPP \"sum\" / SPP \"average trade\" |",
        "|---|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    plus_counts = np.zeros(n, dtype=int)
    plus_no_transfer = np.zeros(n, dtype=int)
    spp_transfer_steps = np.zeros(n, dtype=int)
    plus_steps_total = 0
    plus_steps_dead = 0
    for k in range(5):
        st = [r["steps"][k] for r in rows]
        sign = np.array([s["sum"]["champ_pos"] for s in st])
        plc = np.array([s["sum"]["champ_place"] for s in st])
        gpl = np.array([s["sum"]["g_placem"] for s in st])
        gap = np.array([s["sum"]["g_gapm"] for s in st])
        ver = [verdict(p, lo_s, hi_s) for p in gpl]
        ver_a = [verdict(s["avg"]["g_placem"], lo_a, hi_a) for s in st]
        keep_sp = np.array([v == "перенос" for v in ver])
        keep_sp_a = np.array([v == "перенос" for v in ver_a])
        alive = ALIVE[k]
        ok_sc = int(np.sum(sign == int(alive)))
        ok_sp = int(np.sum(keep_sp == alive))
        ok_sa = int(np.sum(keep_sp_a == alive))
        plus_counts += sign
        if not alive:
            plus_no_transfer += sign
        spp_transfer_steps += keep_sp.astype(int)
        plus_steps_total += int(sign.sum())
        if not alive:
            plus_steps_dead += int(sign.sum())
        L.append(
            f"| {STEP_NAME[k]} | {TEST_PIECE_TYPE[k]} | {STEP_ALPHA[k]:+.2f} | {'yes' if alive else 'no'} | "
            f"{sign.sum()}/{n} | {plc.mean():.0f} | {gpl.mean():.0f} | {gap.mean():+.1f} | "
            f"{ver.count('перенос')} / {ver.count('нет вывода')} / {ver.count('инверсия')} | {ok_sc} / {ok_sp} / {ok_sa} |"
        )
    L += ["", "### Count of \"pluses out of five\" on a ribbon", ""]
    dist = [int(np.sum(plus_counts == k)) for k in range(6)]
    L.append("Ribbons with k positive steps out of five: " + ", ".join(f"k={k}: {d}" for k, d in enumerate(dist)) + ".")
    L.append(
        f"Mean pluses per ribbon {plus_counts.mean():.2f}. Ribbons with four pluses or more: "
        f"{int(np.sum(plus_counts >= 4))}/{n}."
    )
    L.append(
        f"Of {plus_steps_total} positive steps, those on steps with no live order: {plus_steps_dead} "
        f"({100 * plus_steps_dead / plus_steps_total:.0f}%)."
    )
    L.append(
        f"Pluses with no live order per ribbon: mean {plus_no_transfer.mean():.2f}, minimum {plus_no_transfer.min()}, "
        f"maximum {plus_no_transfer.max()}. Ribbons where, among ≥4 pluses, two or more pluses have no order: "
        f"{int(np.sum((plus_counts >= 4) & (plus_no_transfer >= 2)))}/{n}."
    )
    L.append(
        f"Steps with the verdict transfer per ribbon (\"sum\" rule): mean {spp_transfer_steps.mean():.2f}; "
        f"a ribbon has two live steps."
    )

    L += [
        "",
        "## 2. Combinatorial splits",
        "",
        "The \"below the middle\" bit is the chosen candidate's place on test < 50. \"In the minus\" is the chosen sum ≤ 0. "
        "The top slice is the top 10% by the train sum; its gap and percentile are at equal trade frequency: \"gap < 0\" and \"percentile < 50\".",
        "",
        "### One piece on test (6 splits per ribbon)",
        "",
        "| Test piece | Piece type | Below the middle | In the minus | Top-slice gap < 0 (at equal frequency) | Top-slice percentile < 50 | Chosen percentile (ordinary) | Top-slice percentile (at equal frequency) | Top-slice gap (at equal frequency) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    piece_type = ["strong", "strong", "weak", "strong, inversion", "medium", "strong"]
    loo_bits, loo_gap_neg, loo_loss = [], [], []
    for h in range(6):
        cells = [r["loo"][h] for r in rows]
        below = np.array([x["champ_place"] < 50 for x in cells])
        loss = np.array([x["champ_sum"] <= 0 for x in cells])
        gneg = np.array([x["g_gapm"] < 0 for x in cells])
        gp50 = np.array([x["g_placem"] < 50 for x in cells])
        L.append(
            f"| {h + 1} | {piece_type[h]} | {below.sum()}/{n} | {loss.sum()}/{n} | {gneg.sum()}/{n} | {gp50.sum()}/{n} | "
            f"{np.mean([x['champ_place'] for x in cells]):.0f} | {np.mean([x['g_placem'] for x in cells]):.0f} | "
            f"{np.mean([x['g_gapm'] for x in cells]):+.1f} |"
        )
    pbo_loo = np.array([np.mean([x["champ_place"] < 50 for x in r["loo"]]) for r in rows])
    loss_loo = np.array([np.mean([x["champ_sum"] <= 0 for x in r["loo"]]) for r in rows])
    gneg_loo = np.array([np.mean([x["g_gapm"] < 0 for x in r["loo"]]) for r in rows])
    L += [
        "",
        f"Share of splits below the middle per ribbon: mean {pbo_loo.mean():.2f}, SD across ribbons {pbo_loo.std():.2f}, "
        f"from {pbo_loo.min():.2f} to {pbo_loo.max():.2f}. In the minus: mean {loss_loo.mean():.2f}. "
        f"Group gap < 0: mean {gneg_loo.mean():.2f}.",
        "",
        "### Three pieces on test (20 splits per ribbon)",
        "",
    ]
    pbo_tr = np.array([np.mean([x["champ_place"] < 50 for x in r["triples"]]) for r in rows])
    loss_tr = np.array([np.mean([x["champ_sum"] <= 0 for x in r["triples"]]) for r in rows])
    gneg_tr = np.array([np.mean([x["g_gapm"] < 0 for x in r["triples"]]) for r in rows])
    gp50_tr = np.array([np.mean([x["g_placem"] < 50 for x in r["triples"]]) for r in rows])
    L += [
        "| Quantity | Mean across ribbons | SD across ribbons | Minimum | Maximum |",
        "|---|---:|---:|---:|---:|",
        f"| Below the middle (bit, as in PBO) | {pbo_tr.mean():.2f} | {pbo_tr.std():.2f} | {pbo_tr.min():.2f} | {pbo_tr.max():.2f} |",
        f"| In the minus (probability of a loss) | {loss_tr.mean():.2f} | {loss_tr.std():.2f} | {loss_tr.min():.2f} | {loss_tr.max():.2f} |",
        f"| Group gap < 0 | {gneg_tr.mean():.2f} | {gneg_tr.std():.2f} | {gneg_tr.min():.2f} | {gneg_tr.max():.2f} |",
        f"| Group place < 50 | {gp50_tr.mean():.2f} | {gp50_tr.std():.2f} | {gp50_tr.min():.2f} | {gp50_tr.max():.2f} |",
        "",
    ]
    # agreement of the bit and the group on every split
    def agree(cells_all):
        b = np.array([x["champ_place"] < 50 for x in cells_all])
        g = np.array([x["g_gapm"] < 0 for x in cells_all])
        return int(np.sum(b & g)), int(np.sum(b & ~g)), int(np.sum(~b & g)), int(np.sum(~b & ~g)), len(cells_all)

    a_loo = agree([x for r in rows for x in r["loo"]])
    a_tr = agree([x for r in rows for x in r["triples"]])
    L += [
        "### Agreement of the \"below the middle\" bit and the group gap (every split of every ribbon)",
        "",
        "| Splits | Bit and gap<0 | Bit, gap≥0 | No bit, gap<0 | No bit, gap≥0 | Total |",
        "|---|---:|---:|---:|---:|---:|",
        f"| One piece | {a_loo[0]} | {a_loo[1]} | {a_loo[2]} | {a_loo[3]} | {a_loo[4]} |",
        f"| Three pieces | {a_tr[0]} | {a_tr[1]} | {a_tr[2]} | {a_tr[3]} | {a_tr[4]} |",
        "",
        "Disagreements: \"bit, gap≥0\" — the champion is below the middle and the group is above a random set; "
        "\"no bit, gap<0\" — the champion is above the middle and the group is worse than a random set.",
        "",
    ]
    return "\n".join(L) + "\n"


def main() -> None:
    thr = json.loads((HERE / "spp_worlds_study" / "thresholds.json").read_text(encoding="utf-8"))["thresholds"]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seeds = [SEED0 + i for i in range(N_RIBBONS)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(ribbon, seeds, chunksize=1))
    print(f"ribbons: {len(rows)} in {time.time() - t0:.0f} s", flush=True)
    (OUT / "ribbons.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    (OUT / "report.md").write_text(report(rows, thr), encoding="utf-8")
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
