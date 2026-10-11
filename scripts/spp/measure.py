"""SPP on a table you already have.

Each row is one candidate. ``train_score`` is the number the selection rule
saw on the training window. ``test_values`` is what that same row did on the
next window. Rows stay in the same order. The chosen set is not re-picked
on the test window.

Place runs from 0 at the worst test value to just under 100 at the best.
A group's place is the median of its members. Gap is the group's mean minus
the mean of same-sized draws without replacement. z divides that gap by the
spread of those draws. It is a Monte Carlo scale, not a test and not a p-value.

The fields match the live-fold tables: champion, positive share, several
fractions, several rules, deciles, and top-k overlap. They do not include
parameter margins, zones, or a genetic-search universe.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compare_rules",
    "decile_ladder",
    "decile_table",
    "gap_against_random",
    "gap_matched",
    "group_median_place",
    "places",
    "places_matched",
    "selection_profile",
    "selection_report",
    "spearman",
    "spearman_matched",
    "top_index",
    "top_overlap",
]

FRACTIONS = (0.01, 0.05, 0.10, 0.30)


def _column(name: str, values) -> np.ndarray:
    column = np.asarray(values, dtype=np.float64)
    if column.ndim != 1:
        raise ValueError(f"{name} must be one column of numbers")
    if column.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(column).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return column


def _pair(train_score, test_values) -> tuple[np.ndarray, np.ndarray]:
    train = _column("train_score", train_score)
    test = _column("test_values", test_values)
    if len(train) != len(test):
        raise ValueError(
            f"train_score has {len(train)} rows, test_values has {len(test)}"
        )
    return train, test


def _fraction_size(fraction: float, n: int) -> int:
    if not np.isfinite(fraction) or fraction <= 0 or fraction > 1:
        raise ValueError("fraction must be greater than 0 and at most 1")
    return max(1, int(round(float(fraction) * n)))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        return float("nan")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return float("nan")
    if np.unique(left).size < 2 or np.unique(right).size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(left, kind="mergesort"), kind="mergesort").astype(np.float64)
    rb = np.argsort(np.argsort(right, kind="mergesort"), kind="mergesort").astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def places(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    rank = np.empty(len(values), dtype=np.float64)
    rank[order] = np.linspace(0, 100, len(values), endpoint=False)
    return rank


def top_index(score: np.ndarray, fraction: float) -> np.ndarray:
    column = _column("score", score)
    k = _fraction_size(fraction, len(column))
    return np.argsort(column, kind="mergesort")[-k:]


def group_median_place(values: np.ndarray, chosen: np.ndarray) -> float:
    return float(np.median(places(values)[_indexes(chosen, len(values))]))


def _indexes(chosen, n: int) -> np.ndarray:
    picked = np.asarray(chosen)
    if picked.dtype == np.bool_:
        raise ValueError("chosen must be integer row numbers, not a true/false mask")
    if picked.ndim != 1 or picked.size == 0:
        raise ValueError("chosen is empty")
    if not np.issubdtype(picked.dtype, np.integer):
        if not np.isfinite(picked.astype(np.float64)).all():
            raise ValueError("chosen row numbers must be finite integers")
        if not np.all(picked.astype(np.float64) == np.trunc(picked.astype(np.float64))):
            raise ValueError("chosen row numbers must be integers")
    picked = picked.astype(np.int64, copy=False)
    if len(np.unique(picked)) != len(picked) or picked.min() < 0 or picked.max() >= n:
        raise ValueError("chosen row numbers must be unique and inside the column")
    return picked


def gap_against_random(
    values: np.ndarray,
    chosen: np.ndarray,
    draws: int = 300,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Gap and z of ``chosen`` against same-sized draws from ``values``."""
    cloud = _column("values", values)
    picked = _indexes(chosen, len(cloud))
    if not isinstance(draws, (int, np.integer)) or int(draws) < 2:
        raise ValueError("draws must be an integer of at least 2")
    draws = int(draws)
    if rng is None:
        rng = np.random.default_rng(0)
    k = len(picked)
    selected = cloud[picked]
    selected_mean = float(selected.mean())
    sample_mean = np.empty(draws)
    sample_positive = np.empty(draws)
    n = len(cloud)
    for i in range(draws):
        draw = cloud[rng.choice(n, size=k, replace=False)]
        sample_mean[i] = draw.mean()
        sample_positive[i] = np.mean(draw > 0)
    random_mean = float(sample_mean.mean())
    gap = selected_mean - random_mean
    std = float(sample_mean.std())
    z = float("nan") if k == n or std < 1e-8 else gap / std
    return {
        "selected_mean": selected_mean,
        "selected_median": float(np.median(selected)),
        "selected_positive": float(np.mean(selected > 0)),
        "universe_mean": float(cloud.mean()),
        "universe_positive": float(np.mean(cloud > 0)),
        "random_mean": random_mean,
        "random_sd": std,
        "random_p05": float(np.quantile(sample_mean, 0.05)),
        "random_p50": float(np.quantile(sample_mean, 0.50)),
        "random_p95": float(np.quantile(sample_mean, 0.95)),
        "random_positive": float(sample_positive.mean()),
        "gap": gap,
        "gap_exact": selected_mean - float(cloud.mean()),
        "z": z,
        "place": float(np.median(places(cloud)[picked])),
        "random_ge_selected": float(np.mean(sample_mean >= selected_mean)),
        "draws": float(draws),
    }


def _slots(values: np.ndarray, match_on, bins: int) -> tuple[np.ndarray, int]:
    weight = _column("match_on", match_on)
    if len(weight) != len(values):
        raise ValueError(f"values has {len(values)} rows, match_on has {len(weight)}")
    if not isinstance(bins, (int, np.integer)) or int(bins) < 1:
        raise ValueError("bins must be an integer of at least 1")
    bins = int(bins)
    edges = np.quantile(weight, np.linspace(0.0, 1.0, bins + 1))
    slot = np.clip(np.searchsorted(edges, weight, side="right") - 1, 0, bins - 1)
    return slot, bins


def spearman_matched(train_score, test_values, match_on, bins: int = 10) -> float:
    """Spearman between ``train_score`` and ``test_values`` inside quantile bins of ``match_on``.

    Same bins as ``gap_matched`` and ``places_matched``. The result is the mean of the
    per-bin coefficients weighted by bin size; bins with fewer than three rows or a
    constant column are skipped. This is the rank-link counterpart of the matched gap:
    a row with many trades is compared only with rows that trade about as often.
    """
    train, test = _pair(train_score, test_values)
    slot, bins = _slots(test, match_on, bins)
    rhos, weights = [], []
    for k in range(bins):
        members = np.flatnonzero(slot == k)
        if members.size < 3:
            continue
        r = spearman(train[members], test[members])
        if np.isfinite(r):
            rhos.append(r)
            weights.append(members.size)
    return float(np.average(rhos, weights=weights)) if rhos else float("nan")


def places_matched(values: np.ndarray, match_on: np.ndarray, bins: int = 10) -> np.ndarray:
    """Place of each row among rows with a similar ``match_on`` (0 worst, just under 100 best).

    Same quantile bins as ``gap_matched``. A group's matched place is the median of
    these, the rank counterpart of the matched gap.
    """
    cloud = _column("values", values)
    slot, bins = _slots(cloud, match_on, bins)
    out = np.empty(len(cloud), dtype=np.float64)
    for k in range(bins):
        members = np.flatnonzero(slot == k)
        if members.size:
            out[members] = places(cloud[members])
    return out


def gap_matched(
    values: np.ndarray,
    chosen: np.ndarray,
    match_on: np.ndarray,
    bins: int = 10,
) -> dict[str, float]:
    """Gap against a random draw with the same mix of ``match_on`` (trade count).

    Rows are put into ``bins`` quantile bins of ``match_on``. A random draw that
    takes the same number of rows from each bin as the chosen slice has, in
    expectation, the mean of the per-bin means weighted by how many chosen rows
    fall in each bin. The gap is the chosen mean minus that expectation, exact,
    no draws. Use it when the sum on the test window depends on how often a row
    trades: a plain gap then rewards or punishes frequency, not the selection.
    """
    cloud = _column("values", values)
    picked = _indexes(chosen, len(cloud))
    slot, bins = _slots(cloud, match_on, bins)
    bin_mean = np.zeros(bins)
    for k in range(bins):
        members = slot == k
        if members.any():
            bin_mean[k] = cloud[members].mean()
    expected = float(bin_mean[slot[picked]].mean())
    selected = float(cloud[picked].mean())
    return {
        "selected_mean": selected,
        "matched_expected": expected,
        "gap_matched": selected - expected,
        "gap_plain": selected - float(cloud.mean()),
    }


def random_gap(
    chosen_mean: float,
    cloud: np.ndarray,
    k: int,
    rng: np.random.Generator,
    draws: int = 300,
) -> tuple[float, float]:
    """Gap and z as the catalog report prints them (random draws, old edge cases kept).

    The article itself uses the exact gap (``gap_exact``) and does not use z.
    """
    n = len(cloud)
    sample = np.empty(draws)
    for i in range(draws):
        sample[i] = cloud[rng.choice(n, size=k, replace=False)].mean()
    gap = chosen_mean - float(sample.mean())
    std = float(sample.std())
    z = gap / std if std else 0.0
    return gap, z


def selection_report(
    train_score,
    test_values,
    fraction: float = 0.10,
    draws: int = 300,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Freeze the top ``fraction`` by ``train_score`` and read it on ``test_values``."""
    train, test = _pair(train_score, test_values)
    chosen = top_index(train, fraction)
    champion = int(top_index(train, 1.0 / len(train))[0])
    out = gap_against_random(test, chosen, draws=draws, rng=rng)
    out["fraction"] = float(fraction)
    out["n"] = float(len(test))
    out["k"] = float(len(chosen))
    out["champion_value"] = float(test[champion])
    out["champion_place"] = float(places(test)[champion])
    return out


def selection_profile(
    train_score,
    test_values,
    fractions: tuple[float, ...] = FRACTIONS,
    draws: int = 10_000,
    rng: np.random.Generator | None = None,
) -> list[dict[str, float]]:
    """The live-fold rows: 1%, 5%, 10%, and 30%, unless you pass other fractions."""
    if rng is None:
        rng = np.random.default_rng(0)
    return [
        selection_report(train_score, test_values, fraction=fraction, draws=draws, rng=rng)
        for fraction in fractions
    ]


def compare_rules(
    rules: dict[str, np.ndarray],
    test_values,
    fractions: tuple[float, ...] = FRACTIONS,
    draws: int = 10_000,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """One test column, several training rules. Same shape as a selector table."""
    if rng is None:
        rng = np.random.default_rng(0)
    rows = []
    for name, score in rules.items():
        for row in selection_profile(score, test_values, fractions, draws, rng):
            labeled = dict(row)
            labeled["rule"] = name
            rows.append(labeled)
    return rows


def decile_ladder(train_score, test_values, bins: int = 10) -> np.ndarray:
    """Mean test value from the top training slice down to the bottom."""
    return np.array([row["mean"] for row in decile_table(train_score, test_values, bins)])


def decile_table(train_score, test_values, bins: int = 10) -> list[dict[str, float]]:
    """Deciles of the training score. Decile 1 is the top of that score."""
    train, test = _pair(train_score, test_values)
    if not isinstance(bins, (int, np.integer)) or int(bins) < 2:
        raise ValueError("bins must be an integer of at least 2")
    bins = int(bins)
    if len(train) < bins:
        raise ValueError(f"need at least {bins} rows for {bins} deciles, got {len(train)}")
    order = np.argsort(train, kind="mergesort")
    parts = list(reversed(np.array_split(order, bins)))
    rows = []
    for number, part in enumerate(parts, start=1):
        vals = test[part]
        rows.append({
            "decile": float(number),
            "n": float(len(part)),
            "mean": float(vals.mean()),
            "median": float(np.median(vals)),
            "positive_share": float(np.mean(vals > 0)),
        })
    return rows


def top_overlap(train_score, test_values, ks: tuple[int, ...] = (10, 100)) -> list[dict[str, float]]:
    """Share of names that are in the top k on both windows. Chance share is k / n."""
    train, test = _pair(train_score, test_values)
    n = len(train)
    train_order = np.argsort(train, kind="mergesort")[::-1]
    test_order = np.argsort(test, kind="mergesort")[::-1]
    rows = []
    for k in ks:
        kk = int(k)
        if kk < 1 or kk > n:
            continue
        shared = len(set(train_order[:kk].tolist()) & set(test_order[:kk].tolist()))
        rows.append({
            "k": float(kk),
            "overlap": float(shared),
            "overlap_share": shared / kk,
            "chance_share": kk / n,
        })
    return rows
