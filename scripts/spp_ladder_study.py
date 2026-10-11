"""Train-decile ladder and the sum / average-trade fork on the five configurations.

The same worlds as spp_worlds_study (seeds 1000-1029, configuration 5 offset by
5000). For each configuration:

- ladder: mean test sum across the ten slices of the train "sum" score
  (slice 1 is the top of training), averaged over 30 worlds;
- fork: what the "sum" and "average trade" rules select — trade count of the
  top 10% and the place of the chosen names among those who traded as often.

    python scripts/spp_ladder_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spp_catalog_six as c  # noqa: E402
from spp.measure import decile_table, places_matched, top_index  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "spp_ladder_study"
SEED_CFG = 1000
N = 30
SECOND_TYPE_OFFSET = 5000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for idx, (_name, _intended, a1, s1, a2, s2) in enumerate(c.CATALOG, start=1):
        off = SECOND_TYPE_OFFSET if idx == 5 else 0
        ladders, ladders_pos, fam, fork = [], [], [], {"sum": [], "avg": []}
        trades_all = []
        for i in range(N):
            w = c.build_pair(SEED_CFG + i + off, a1, s1, a2, s2)
            tn = w["test_net"]
            tab = decile_table(w["train_net"], tn, 10)
            ladders.append([r["mean"] for r in tab])
            ladders_pos.append([r["positive_share"] for r in tab])
            fam.append(float(tn.mean()))
            trades_all.append(float(w["trades"].mean()))
            pm = places_matched(tn, w["trades"])
            for rule, score in (("sum", w["train_net"]), ("avg", w["train_avg"])):
                chosen = top_index(score, 0.10)
                fork[rule].append({
                    "trades": float(w["trades"][chosen].mean()),
                    "mean_test": float(tn[chosen].mean()),
                    "pos": float(np.mean(tn[chosen] > 0)),
                    "placem": float(np.median(pm[chosen])),
                })
        result[f"cfg{idx}"] = {
            "ladder_mean": np.mean(ladders, axis=0).tolist(),
            "ladder_pos": np.mean(ladders_pos, axis=0).tolist(),
            "family_mean": float(np.mean(fam)),
            "trades_mean": float(np.mean(trades_all)),
            "fork": {r: {k: float(np.mean([d[k] for d in v])) for k in v[0]} for r, v in fork.items()},
        }
        print(idx, "done", flush=True)
    (OUT / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# Decile ladder and the rule fork", "", f"Means over {N} worlds per configuration, seeds {SEED_CFG}–{SEED_CFG + N - 1}.", ""]
    for key, v in result.items():
        L += [f"## {key}", "", f"Family: {v['family_mean']:+.1f}, mean trade count {v['trades_mean']:.0f}.", ""]
        L.append("| Train slice | " + " | ".join(str(d) for d in range(1, 11)) + " |")
        L.append("|---|" + "---:|" * 10)
        L.append("| Mean sum | " + " | ".join(f"{x:+.0f}" for x in v["ladder_mean"]) + " |")
        L.append("")
        L += ["| Rule | trades in the top 10% | mean sum | in the plus | place at equal frequency |", "|---|---:|---:|---:|---:|"]
        for r, t in v["fork"].items():
            L.append(f"| {r} | {t['trades']:.0f} | {t['mean_test']:+.1f} | {t['pos']:.2f} | {t['placem']:.0f} |")
        L.append("")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(OUT / "report.md")


if __name__ == "__main__":
    main()
