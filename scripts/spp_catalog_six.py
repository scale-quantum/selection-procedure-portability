"""A catalog of five transitions, frequencies of random transfer, and six pieces.

A second calibration of the shift constants relative to spp_catalog_five.py.
The first version was a device with an almost binary piece level: weak and
strong pieces put 0% or 100% of candidates in the plus. Because of that the
reading was incomplete in two places.

First: the transition "weak after strong" at a zero piece level handed the
whole sum to the trade frequency (the link of trade count with the sum reached
−0.82), and the rank link of the sum itself fell almost to zero (0.12) — not
because order disappeared, but because at 0% plus candidates on the piece the
sum is entirely determined by who took fewer trades. The average trade on such
a piece shows transfer almost alone, and that looked like an artificially
arranged case rather than an honest boundary of the method.

Second: the "random transfer" transition on the "medium" piece was not actually
medium in the plus share — 83–92% of candidates in the plus, with a claimed
random rank link, is closer to the "weak strong" type, only less extreme.
There was no piece at all where the plus share is near one half — and that is
the piece closest to what a real distribution shows on a live asset with random
transfer (live window 2/1 of the article: 59.3% plus on test, Spearman of
IS metrics with OOS Net 0.03–0.08).

Below, three shift constants are recomputed so that:
- the weak piece stays weak (plus share <=25%, mean sum < 0
  under the same market_label tags), but not zero, and the link of trade count
  with the sum does not exceed about 0.5 in absolute value;
- the flipped strong piece stays strong (plus share >=85%), but
  with the same softened link of trade count with the sum;
- the "medium" piece of the random transition is a piece with a plus share
  near one half, not near 90%.

The fit is scripts/_calibrate_shift.py and scripts/_calibrate_shift2.py (they
are not in the article pipeline; they are kept as a working calibration log).

    python scripts/spp_catalog_six.py
    python scripts/spp_catalog_six.py --quick
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from spp.measure import places, random_gap, spearman

N_OPP = 4_000
N_CLUSTERS = 50
PER = 50
SEED = 41
N_SEEDS = 30
SEED0 = 100
OUT = Path(__file__).resolve().parent / "spp_catalog_six"

# Level shifts are fitted so that the family's mean sum
# lands in a strong / medium / weak piece at a given sign of order.
SHIFT_STRONG = 0.05
# Was -1.45: the weak piece was literally zero in plus share, and the link of
# trade count with the sum reached -0.82, drowning the average-trade signal.
# -0.85 gives a weak but not zero piece (9% plus at alpha 0.80,
# 21.6% at alpha 1.00) and a trade-count link with the sum within -0.5.
SHIFT_WEAK = -0.85
# Was 1.25: the flipped strong piece was 100% plus, and the link of trade
# count with the sum reached 0.86. 0.75 leaves the piece strong under the same
# tags (94% plus) with a more moderate link (0.58).
SHIFT_STRONG_INVERTED = 0.75
# With no shared order of opportunities, the piece level is set by the shift itself.
# Was 0.12: plus share 83-92%, so the "medium" piece was only a less
# extreme strong piece. 0.00 gives a piece whose plus share on a
# single seed fluctuates around one half (46-66% on the seeds checked) —
# closer to a real distribution on an asset with random
# transfer than any of the earlier variants.
SHIFT_FLAT_MEDIUM = 0.00
SHIFT_FLAT_STRONG = 0.60


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -8, 8)))


def draw(chance: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.where(rng.random(chance.shape) < chance, np.int8(1), np.int8(-1))


def take(scores: np.ndarray, n_trades: np.ndarray) -> np.ndarray:
    mask = np.zeros(scores.shape, dtype=bool)
    for i, trades in enumerate(n_trades):
        picked = np.argpartition(scores[i], -int(trades))[-int(trades):]
        mask[i, picked] = True
    return mask


def latent(alpha: float, shift: float, edge: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if abs(alpha) >= 0.999:
        base = np.sign(alpha) * edge
    else:
        noise = rng.normal(0, 1, size=edge.shape).astype(np.float32)
        width = np.sqrt(max(0.0, 1.0 - alpha * alpha))
        base = alpha * edge + width * noise
    return (base + shift).astype(np.float32)


def build_mask(rng: np.random.Generator, edge: np.ndarray):
    n = N_CLUSTERS * PER
    assign = np.repeat(np.arange(N_CLUSTERS), PER)
    exposure = np.clip(rng.normal(0.40, 0.16, size=N_CLUSTERS), 0.05, 0.85).astype(np.float32)
    residual = rng.normal(0, 1, size=(N_CLUSTERS, N_OPP)).astype(np.float32)
    width = np.sqrt(np.clip(1 - exposure ** 2, 0.02, None)).astype(np.float32)
    centers = exposure[:, None] * edge + width[:, None] * residual
    scores = centers[assign] + rng.normal(0, 0.22, size=(n, N_OPP)).astype(np.float32)
    base = rng.uniform(280, 640, size=N_CLUSTERS)
    n_trades = np.clip(base[assign] + rng.normal(0, 16, size=n), 180, 860).astype(np.int32)
    mask = take(scores, n_trades)
    return mask, n_trades.astype(np.float64), assign, exposure


def outcomes_from_latent(mask: np.ndarray, latent_row: np.ndarray, rng: np.random.Generator):
    result = np.where(mask, draw(sigmoid(latent_row), rng), np.int8(0))
    net = result.sum(axis=1).astype(np.float64)
    trades = np.maximum(mask.sum(axis=1), 1).astype(np.float64)
    avg = net / trades
    return net, avg


def build_pair(seed: int, train_alpha: float, train_shift: float, test_alpha: float, test_shift: float):
    rng = np.random.default_rng(seed)
    edge = rng.normal(0.0, 1.0, size=N_OPP).astype(np.float32)
    mask, trades, assign, exposure = build_mask(rng, edge)
    train_lat = latent(train_alpha, train_shift, edge, rng)
    test_lat = latent(test_alpha, test_shift, edge, rng)
    train_net, train_avg = outcomes_from_latent(mask, train_lat, rng)
    test_net, test_avg = outcomes_from_latent(mask, test_lat, rng)
    return {
        "train_net": train_net,
        "train_avg": train_avg,
        "test_net": test_net,
        "test_avg": test_avg,
        "trades": trades,
        "assign": assign,
        "exposure": exposure,
    }


def build_chain(seed: int, specs: list[tuple[float, float]]):
    rng = np.random.default_rng(seed)
    edge = rng.normal(0.0, 1.0, size=N_OPP).astype(np.float32)
    mask, trades, assign, exposure = build_mask(rng, edge)
    nets = []
    avgs = []
    for alpha, shift in specs:
        row = latent(alpha, shift, edge, rng)
        net, avg = outcomes_from_latent(mask, row, rng)
        nets.append(net)
        avgs.append(avg)
    return nets, avgs, trades, assign, exposure


def cloud_bits(net: np.ndarray) -> str:
    q10, q50, q90 = np.quantile(net, [0.10, 0.50, 0.90])
    return (
        f"mean {net.mean():+.1f}, median {q50:+.1f}, spread {net.std():.1f}, "
        f"bottom 10% {q10:+.1f}, top 10% {q90:+.1f}, "
        f"in the plus {100 * np.mean(net > 0):.1f}%"
    )


def market_label(net: np.ndarray) -> str:
    mean = float(net.mean())
    pos = float(np.mean(net > 0))
    if mean > 40 and pos >= 0.85:
        return "strong"
    if mean < 0 and pos <= 0.25:
        return "weak"
    return "medium"


def transfer_label(rho: float) -> str:
    if rho >= 0.65:
        return "strong"
    if rho >= 0.35:
        return "moderate"
    if rho >= -0.20:
        return "random"
    return "inversion"


def top_idx(score: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, int(round(frac * len(score))))
    return np.argsort(score, kind="mergesort")[-k:]


def champion(score: np.ndarray) -> int:
    """Best by the score; on tied sums, the last row, as top_idx and spp.top_index."""
    return int(np.argsort(score, kind="mergesort")[-1])


def cluster_ids_by_train(score: np.ndarray, assign: np.ndarray) -> list[int]:
    means = []
    for cluster in range(N_CLUSTERS):
        members = np.flatnonzero(assign == cluster)
        means.append((float(score[members].mean()), cluster))
    means.sort(reverse=True)
    return [cluster for _, cluster in means]


def member_mask(assign: np.ndarray, cluster: int) -> np.ndarray:
    return assign == cluster


def one_candidate(title: str, score: np.ndarray, test_net: np.ndarray, test_avg: np.ndarray, trades: np.ndarray, assign: np.ndarray) -> list[str]:
    champ = champion(score)
    place_net = float(places(test_net)[champ])
    place_avg = float(places(test_avg)[champ])
    own = member_mask(assign, int(assign[champ]))
    own_place = float(np.median(places(test_net)[own]))
    ranked = cluster_ids_by_train(score, assign)
    own_id = int(assign[champ])
    second = ranked[1] if ranked[0] == own_id else ranked[0]
    second_members = member_mask(assign, second)
    second_place = float(np.median(places(test_net)[second_members]))
    return [
        title,
        (
            f"  One best: sum on test {test_net[champ]:+.1f}, quality {test_avg[champ]:+.3f}, "
            f"trades {trades[champ]:.0f}. Place by sum {place_net:.0f}, by quality {place_avg:.0f} "
            f"(100 is the top of the family). Family {test_net.mean():+.1f}."
        ),
        (
            f"  Its group of similar rows: mean sum {test_net[own].mean():+.1f}, "
            f"median place {own_place:.0f}, in the plus {100 * np.mean(test_net[own] > 0):.0f}%. "
            f"Gap of the best from its own group {test_net[champ] - test_net[own].mean():+.1f}."
        ),
        (
            f"  Second group by the same training: mean sum {test_net[second_members].mean():+.1f}, "
            f"median place {second_place:.0f}, in the plus {100 * np.mean(test_net[second_members] > 0):.0f}%."
        ),
    ]


def group_lines(title: str, score: np.ndarray, test_net: np.ndarray, rng: np.random.Generator) -> list[str]:
    lines = [title]
    rank = places(test_net)
    for frac, label in ((0.01, "top 1%"), (0.10, "top 10%")):
        chosen = top_idx(score, frac)
        selected = test_net[chosen]
        gap, z = random_gap(float(selected.mean()), test_net, len(chosen), rng)
        lines.append(
            f"  {label}: sum {selected.mean():+.1f}, median place {float(np.median(rank[chosen])):.0f}, "
            f"in the plus {100 * np.mean(selected > 0):.0f}%, "
            f"above random sets of the same size by {gap:+.1f} (z {z:.1f})."
        )
    return lines


def decile_lines_with_trades(score: np.ndarray, test_net: np.ndarray, test_avg: np.ndarray, trades: np.ndarray) -> list[str]:
    order = np.argsort(score, kind="mergesort")
    bins = np.array_split(order, 10)
    lines = ["  Ladder of tenths, from the top of training to the bottom:"]
    for i, bucket in enumerate(reversed(bins), start=1):
        lines.append(
            f"    {i}: sum {test_net[bucket].mean():+.1f}, quality {test_avg[bucket].mean():+.3f}, "
            f"in the plus {100 * np.mean(test_net[bucket] > 0):.0f}%, mean trades {trades[bucket].mean():.0f}."
        )
    return lines


def transition_report(name: str, intended: str, world: dict, rng: np.random.Generator) -> list[str]:
    train_net = world["train_net"]
    test_net = world["test_net"]
    train_avg = world["train_avg"]
    test_avg = world["test_avg"]
    trades = world["trades"]
    assign = world["assign"]
    rho_net = spearman(train_net, test_net)
    rho_avg = spearman(train_avg, test_avg)
    rho_trades = spearman(trades, test_net)
    overlap = len(set(top_idx(train_net, 0.10).tolist()) & set(top_idx(test_net, 0.10).tolist()))
    overlap_avg = len(set(top_idx(train_avg, 0.10).tolist()) & set(top_idx(test_avg, 0.10).tolist()))
    k = max(1, int(round(0.10 * len(train_net))))
    lines = [
        f"## {name}",
        "",
        f"Intended: {intended}.",
        f"Obtained: training {market_label(train_net)}, test {market_label(test_net)}, "
        f"transfer by trade quality {transfer_label(rho_avg)}.",
        "",
        f"Training: {cloud_bits(train_net)}.",
        f"Test: {cloud_bits(test_net)}.",
        f"Mean trades {trades.mean():.0f}, spread {trades.std():.0f}.",
        f"Rank link of the sum {rho_net:.2f}. Rank link of quality {rho_avg:.2f}. "
        f"Link of trade count and sum on test {rho_trades:.2f}.",
        f"Names in common in the top 10% by sum: {100 * overlap / k:.0f}%. By quality: {100 * overlap_avg / k:.0f}%.",
        "",
        *one_candidate("Best by sum on training.", train_net, test_net, test_avg, trades, assign),
        *group_lines("Group selected by sum.", train_net, test_net, rng),
        *decile_lines_with_trades(train_net, test_net, test_avg, trades),
        "",
        *one_candidate("Best by trade quality on training.", train_avg, test_net, test_avg, trades, assign),
        *group_lines("Group selected by trade quality.", train_avg, test_net, rng),
        *decile_lines_with_trades(train_avg, test_net, test_avg, trades),
        "",
    ]
    return lines


def snapshot(world: dict) -> dict:
    train_net = world["train_net"]
    test_net = world["test_net"]
    train_avg = world["train_avg"]
    test_avg = world["test_avg"]
    trades = world["trades"]
    assign = world["assign"]
    out = {
        "rho_avg": spearman(train_avg, test_avg),
        "rho_net": spearman(train_net, test_net),
        "test_mean": float(test_net.mean()),
        "test_pos": float(np.mean(test_net > 0)),
    }
    for key, score in (("sum", train_net), ("avg", train_avg)):
        champ = champion(score)
        place = places(test_net)
        out[f"{key}_place"] = float(place[champ])
        out[f"{key}_pnl"] = float(test_net[champ])
        out[f"{key}_pos"] = int(test_net[champ] > 0)
        own = member_mask(assign, int(assign[champ]))
        out[f"{key}_own_place"] = float(np.median(place[own]))
        ranked = cluster_ids_by_train(score, assign)
        own_id = int(assign[champ])
        second = ranked[1] if ranked[0] == own_id else ranked[0]
        out[f"{key}_second_place"] = float(np.median(place[member_mask(assign, second)]))
        chosen = top_idx(score, 0.10)
        out[f"{key}_group_place"] = float(np.median(place[chosen]))
        out[f"{key}_group_mean"] = float(test_net[chosen].mean())
        out[f"{key}_group_pos"] = float(np.mean(test_net[chosen] > 0))
        gap, _ = random_gap(float(test_net[chosen].mean()), test_net, len(chosen), np.random.default_rng(0), draws=80)
        out[f"{key}_gap"] = gap
    out["trades_rho"] = spearman(trades, test_net)
    return out


def freq_report(title: str, rows: list[dict], seeds: range | None = None) -> list[str]:
    n = len(rows)
    if seeds is None:
        seed_text = f"{SEED0}…{SEED0 + n - 1}"
    else:
        seed_text = f"{seeds.start}…{seeds.stop - 1}"
    def share(pred) -> str:
        return f"{100 * np.mean([pred(r) for r in rows]):.0f}%"

    def dist(key: str) -> str:
        vals = np.array([r[key] for r in rows], dtype=float)
        q10, q50, q90 = np.quantile(vals, [0.10, 0.50, 0.90])
        return f"mean {vals.mean():.0f}, median {q50:.0f}, bottom 10% of worlds {q10:.0f}, top 10% of worlds {q90:.0f}"

    lines = [
        f"## {title}",
        "",
        f"Repeats of this world type: {n}. Seeds {seed_text}.",
        f"Quality rank link: mean {np.mean([r['rho_avg'] for r in rows]):.2f}. "
        f"Sum rank link: mean {np.mean([r['rho_net'] for r in rows]):.2f}. "
        f"Link of trades and test sum: mean {np.mean([r['trades_rho'] for r in rows]):.2f}.",
        f"Test: mean family sum {np.mean([r['test_mean'] for r in rows]):+.1f}, "
        f"in the plus on average {100 * np.mean([r['test_pos'] for r in rows]):.0f}%.",
        "",
        "Best by sum.",
        f"  In the plus on test: {share(lambda r: r['sum_pos'])}.",
        f"  Place not below the middle: {share(lambda r: r['sum_place'] >= 50)}.",
        f"  Place in the top 10%: {share(lambda r: r['sum_place'] >= 90)}.",
        f"  Place in the bottom 10%: {share(lambda r: r['sum_place'] <= 10)}.",
        f"  Place: {dist('sum_place')}.",
        f"  Its group: {dist('sum_own_place')}. Second group: {dist('sum_second_place')}.",
        f"  Top 10% by sum, place: {dist('sum_group_place')}. "
        f"Mean gap versus random {np.mean([r['sum_gap'] for r in rows]):+.1f}. "
        f"In the plus on average {100 * np.mean([r['sum_group_pos'] for r in rows]):.0f}%.",
        "",
        "Best by trade quality.",
        f"  In the plus on test: {share(lambda r: r['avg_pos'])}.",
        f"  Place not below the middle: {share(lambda r: r['avg_place'] >= 50)}.",
        f"  Place in the top 10%: {share(lambda r: r['avg_place'] >= 90)}.",
        f"  Place in the bottom 10%: {share(lambda r: r['avg_place'] <= 10)}.",
        f"  Place: {dist('avg_place')}.",
        f"  Its group: {dist('avg_own_place')}. Second group: {dist('avg_second_place')}.",
        f"  Top 10% by quality, place: {dist('avg_group_place')}. "
        f"Mean gap versus random {np.mean([r['avg_gap'] for r in rows]):+.1f}. "
        f"In the plus on average {100 * np.mean([r['avg_group_pos'] for r in rows]):.0f}%.",
        "",
        "Disagreement of one name and the neighboring group when selecting by quality.",
        f"  Own group below 30, and the second between 40 and 60: {share(lambda r: r['avg_own_place'] < 30 and 40 <= r['avg_second_place'] <= 60)}.",
        f"  The best itself below 30, and the top 10% between 40 and 60: {share(lambda r: r['avg_place'] < 30 and 40 <= r['avg_group_place'] <= 60)}.",
        f"  The best itself above 90, and the top 10% between 40 and 60: {share(lambda r: r['avg_place'] >= 90 and 40 <= r['avg_group_place'] <= 60)}.",
        "",
    ]
    return lines


def matrix_lines(nets: list[np.ndarray], avgs: list[np.ndarray]) -> list[str]:
    lines = ["Rank link of the sum between pieces:"]
    header = "      " + " ".join(f"{i+1:>6}" for i in range(6))
    lines.append(header)
    for i in range(6):
        row = " ".join(f"{spearman(nets[i], nets[j]):6.2f}" for j in range(6))
        lines.append(f"  {i+1:>3} {row}")
    lines.append("Rank link of trade quality:")
    lines.append(header)
    for i in range(6):
        row = " ".join(f"{spearman(avgs[i], avgs[j]):6.2f}" for j in range(6))
        lines.append(f"  {i+1:>3} {row}")
    return lines


def split_row(train_score: np.ndarray, test_nets: list[np.ndarray], rng: np.random.Generator) -> dict:
    champ = champion(train_score)
    places_one = [float(places(net)[champ]) for net in test_nets]
    summed = np.sum(test_nets, axis=0)
    place_sum = float(places(summed)[champ])
    chosen = top_idx(train_score, 0.10)
    group_places = [float(np.median(places(net)[chosen])) for net in test_nets]
    gap, _ = random_gap(float(summed[chosen].mean()), summed, len(chosen), rng, draws=80)
    return {
        "pnl": float(summed[champ]),
        "family": float(summed.mean()),
        "place": place_sum,
        "mean_place": float(np.mean(places_one)),
        "below": int(place_sum < 50),
        "below_mean": int(np.mean(places_one) < 50),
        "positive": int(summed[champ] > 0),
        "group_place": float(np.mean(group_places)),
        "gap": gap,
    }


CATALOG = [
    (
        "Transition 1. Strong market, then strong again, strong transfer",
        "the scalar sees a plus and does not check whether the top is accidental",
        1.00,
        SHIFT_STRONG,
        0.82,
        SHIFT_STRONG,
    ),
    (
        "Transition 2. Strong market, then weak, strong transfer",
        "the scalar sees a minus and does not see that the top of the family was kept",
        1.00,
        SHIFT_STRONG,
        0.80,
        SHIFT_WEAK,
    ),
    (
        "Transition 3. Weak market, then strong, transfer flipped",
        "the scalar sees a plus and does not see that the best landed at the bottom of the family",
        1.00,
        SHIFT_WEAK,
        -0.72,
        SHIFT_STRONG_INVERTED,
    ),
    (
        "Transition 4. Strong market, then medium, random transfer",
        "order on test is unrelated to training; one best name can land high by chance",
        1.00,
        SHIFT_STRONG,
        0.00,
        SHIFT_FLAT_MEDIUM,
    ),
    (
        "Transition 5. Medium market, then strong, random transfer",
        "one best name can sag while the wide top group stays near the middle",
        0.00,
        SHIFT_FLAT_MEDIUM,
        0.00,
        SHIFT_FLAT_STRONG,
    ),
]

CHAIN = [
    (1.00, SHIFT_STRONG),
    (0.82, SHIFT_STRONG),
    (0.78, SHIFT_WEAK),
    (-0.72, SHIFT_STRONG_INVERTED),
    (0.00, SHIFT_FLAT_MEDIUM),
    (0.00, SHIFT_FLAT_STRONG),
]


def main() -> None:
    quick = "--quick" in sys.argv
    judge = np.random.default_rng(SEED + 7)
    lines: list[str] = [
        "# Catalog of transitions and six pieces",
        "",
        f"Candidates {N_CLUSTERS * PER}. Groups of similar rows {N_CLUSTERS} of {PER}. "
        f"Opportunities {N_OPP}. Illustrative seed {SEED}.",
        "Trade outcome: +1, −1, or a skip. The candidate filter is built from one shared order of opportunities. "
        "The piece level and the sign of that order are set separately.",
        "Place 100 is the top of the family, place 0 is the bottom. The scalar here is the single best by the rule on training. "
        "Its \"comfort\" in a rolling split is a plus on test, not a place.",
        "",
        "## Five separate transitions",
        "",
    ]
    print("catalog", flush=True)
    for name, intended, a1, s1, a2, s2 in CATALOG:
        world = build_pair(SEED, a1, s1, a2, s2)
        lines.extend(transition_report(name, intended, world, judge))
        print(" ", name[:24], market_label(world["train_net"]), "->", market_label(world["test_net"]),
              transfer_label(spearman(world["train_avg"], world["test_avg"])), flush=True)

    if not quick:
        print("frequencies", flush=True)
        rows4 = []
        rows5 = []
        for seed in range(SEED0, SEED0 + N_SEEDS):
            rows4.append(snapshot(build_pair(seed, 1.00, SHIFT_STRONG, 0.00, SHIFT_FLAT_MEDIUM)))
            rows5.append(snapshot(build_pair(seed + 5000, 0.00, SHIFT_FLAT_MEDIUM, 0.00, SHIFT_FLAT_STRONG)))
            print(f"  seed {seed}", flush=True)
        lines.extend(freq_report(
            "Frequencies of transition 4: strong training, medium next piece, random transfer",
            rows4,
            range(SEED0, SEED0 + N_SEEDS),
        ))
        lines.extend(freq_report(
            "Frequencies of transition 5: medium training, strong next piece, random transfer",
            rows5,
            range(SEED0 + 5000, SEED0 + 5000 + N_SEEDS),
        ))

    print("chain", flush=True)
    nets, avgs, trades, assign, _exposure = build_chain(SEED, CHAIN)
    lines.extend([
        "## Six pieces of one ribbon",
        "",
        "Levels as intended: strong, strong, weak, strong flipped, medium, strong. "
        "Neighboring transfers: strong, strong, inversion, random, random. "
        "Distant links were not specified; they are computed.",
        "",
    ])
    for i, net in enumerate(nets, start=1):
        lines.append(f"Piece {i}: {market_label(net)}. {cloud_bits(net)}.")
    lines.append("")
    lines.extend(matrix_lines(nets, avgs))
    lines.append("")
    lines.append("Rolling split, training only on the previous piece.")
    lines.append("")
    happy = 0
    for i in range(5):
        champ = champion(nets[i])
        pnl = float(nets[i + 1][champ])
        happy += int(pnl > 0)
        block = transition_report(
            f"Rolling step {i+1}→{i+2}",
            "read the actual level and rank link, not the tag",
            {
                "train_net": nets[i],
                "train_avg": avgs[i],
                "test_net": nets[i + 1],
                "test_avg": avgs[i + 1],
                "trades": trades,
                "assign": assign,
            },
            judge,
        )
        lines.extend(block)
    lines.append(
        f"The sum scalar is in the plus on {happy} steps out of 5. "
        "This is one draw of the ribbon, not a frequency."
    )
    lines.append("")

    print("splits", flush=True)
    lines.append("## Combinatorial splits of the same six pieces")
    lines.append("")
    lines.append(
        "This is not someone else's full procedure with cleaning and a forbidden gap. "
        "It is the share of splits where the best by the train sum lands below the middle of the test."
    )
    lines.append("")
    lines.append("One piece on test, the other five on training.")
    loo = []
    for hold in range(6):
        train_score = np.sum([nets[i] for i in range(6) if i != hold], axis=0)
        row = split_row(train_score, [nets[hold]], judge)
        loo.append(row)
        rho = spearman(train_score, nets[hold])
        lines.append(
            f"  Test of piece {hold+1}: best {row['pnl']:+.1f} against family {row['family']:+.1f}, "
            f"place {row['place']:.0f}, top 10% place {row['group_place']:.0f}, "
            f"gap versus random {row['gap']:+.1f}, rank link of training and test {rho:.2f}. "
            + ("Below the middle." if row["below"] else "Not below the middle.")
            + (" Formally in the plus." if row["positive"] else " Formally in the minus.")
        )
    lines.append(
        f"Share of splits where the best is below the middle: {np.mean([r['below'] for r in loo]):.2f}. "
        f"Share where it is in the plus: {np.mean([r['positive'] for r in loo]):.2f}."
    )
    lines.append("")
    lines.append("Three pieces on test and three on training. There are 20 such splits; the flip is included separately.")
    cscv = []
    for test_idx in combinations(range(6), 3):
        train_idx = [i for i in range(6) if i not in test_idx]
        train_score = np.sum([nets[i] for i in train_idx], axis=0)
        row = split_row(train_score, [nets[i] for i in test_idx], judge)
        row["test"] = tuple(i + 1 for i in test_idx)
        cscv.append(row)
    lines.append(
        f"  Share where the best's place on the summed test is below 50: {np.mean([r['below'] for r in cscv]):.2f}."
    )
    lines.append(
        f"  Share where the mean place across the three test pieces is below 50: {np.mean([r['below_mean'] for r in cscv]):.2f}."
    )
    lines.append(
        f"  Share where the best's summed test is in the plus: {np.mean([r['positive'] for r in cscv]):.2f}."
    )
    lines.append(
        f"  Mean place of the top 10%: {np.mean([r['group_place'] for r in cscv]):.0f}. "
        f"Mean gap versus random {np.mean([r['gap'] for r in cscv]):+.1f}."
    )
    lines.append("  Splits:")
    for row in cscv:
        label = "+".join(str(i) for i in row["test"])
        lines.append(
            f"    test {label}: best {row['pnl']:+.1f}, family {row['family']:+.1f}, "
            f"sum place {row['place']:.0f}, mean place across pieces {row['mean_place']:.0f}, "
            f"top 10% {row['group_place']:.0f}, gap {row['gap']:+.1f}."
        )
    lines.append("")
    lines.extend([
        "## Live monitoring on the same tests",
        "",
        "There is no separate account ribbon. This is a protocol: on each new piece, look at the frozen best "
        "and the frozen top group relative to the whole family they were trained with.",
        "Two readings that the best's sum does not distinguish.",
        "A weak piece and live transfer: the sum is below zero, the top's place is high. The regime is bad, the selection holds.",
        "A plus on the surface and dead transfer: the sum can be above zero, the place is near the middle or below, "
        "and the gap versus random is near zero. The profit belongs to the regime. Tomorrow's weak piece hits a random selection harder "
        "than today's strong regime produced that profit.",
        "The numbers of these two readings are catalog transition 2 and the frequencies of transitions 4 and 5. "
        "The rolling steps of the ribbon above show one concrete draw of the same logic.",
        "",
    ])

    text = "\n".join(lines) + "\n"
    OUT.mkdir(parents=True, exist_ok=True)
    report_name = "report-quick.md" if quick else "report.md"
    (OUT / report_name).write_text(text, encoding="utf-8")
    print(OUT / report_name)


if __name__ == "__main__":
    main()
