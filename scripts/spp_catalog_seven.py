"""Generator variants that check what spp_catalog_six builds in by construction.

spp_catalog_six: the piece level acts on a candidate's sum only through its trade
count; the trade count is the same on train and test and is set at cluster level.
This is the same layout with three switches; at the default values the worlds
match build_pair from spp_catalog_six (checked by --selftest).

    beta_spread      a second level channel. Each candidate gets a sensitivity
                     beta_i = 1 + beta_spread * u_i (u_i uniform from -1 to 1) and
                     the level shift enters the plus probability as beta_i * shift.
                     Trade count does not affect this, so equal frequency does not
                     remove it. The outcome of a taken opportunity is still shared
                     by everyone who took it (one uniform draw per opportunity).
    test_trade_sigma trade count on test = train count * exp(sigma * eps_i),
                     eps_i independent across candidates. A candidate takes the
                     first opportunities in order, so the masks are nested.
    trade_mode       "cluster" (as in spp_catalog_six: the trade-count base is set
                     on the cluster) or "candidate" (each candidate's trade count
                     is independent, uniform from 280 to 640).

    python scripts/spp_catalog_seven.py --selftest
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spp_catalog_six as c  # noqa: E402

N_OPP = c.N_OPP


def build_mask(rng: np.random.Generator, edge: np.ndarray, n_clusters: int, per: int, trade_mode: str):
    n = n_clusters * per
    assign = np.repeat(np.arange(n_clusters), per)
    exposure = np.clip(rng.normal(0.40, 0.16, size=n_clusters), 0.05, 0.85).astype(np.float32)
    residual = rng.normal(0, 1, size=(n_clusters, N_OPP)).astype(np.float32)
    width = np.sqrt(np.clip(1 - exposure ** 2, 0.02, None)).astype(np.float32)
    centers = exposure[:, None] * edge + width[:, None] * residual
    scores = centers[assign] + rng.normal(0, 0.22, size=(n, N_OPP)).astype(np.float32)
    if trade_mode == "cluster":
        base = rng.uniform(280, 640, size=n_clusters)
        n_trades = np.clip(base[assign] + rng.normal(0, 16, size=n), 180, 860).astype(np.int32)
    elif trade_mode == "candidate":
        n_trades = np.clip(rng.uniform(280, 640, size=n) + rng.normal(0, 16, size=n), 180, 860).astype(np.int32)
    else:
        raise ValueError(trade_mode)
    mask = c.take(scores, n_trades)
    return mask, n_trades, assign, scores


def outcomes(mask: np.ndarray, base_or_lat: np.ndarray, rng: np.random.Generator, shift: float = 0.0, beta: np.ndarray | None = None):
    """Without beta: base_or_lat already contains the shift (as in spp_catalog_six). With beta: the shift enters as beta_i * shift."""
    if beta is None:
        result = np.where(mask, c.draw(c.sigmoid(base_or_lat), rng), np.int8(0))
    else:
        u = rng.random(base_or_lat.shape)
        chance = c.sigmoid(base_or_lat[None, :] + (beta * shift).astype(np.float32)[:, None])
        result = np.where(mask, np.where(u[None, :] < chance, np.int8(1), np.int8(-1)), np.int8(0))
    net = result.sum(axis=1).astype(np.float64)
    trades = np.maximum(mask.sum(axis=1), 1).astype(np.float64)
    return net, net / trades


def build_pair(seed: int, a1: float, s1: float, a2: float, s2: float, *, beta_spread: float = 0.0,
               test_trade_sigma: float = 0.0, trade_mode: str = "cluster", n_clusters: int = 50, n_cand: int = 2500):
    per = n_cand // n_clusters
    rng = np.random.default_rng(seed)
    edge = rng.normal(0.0, 1.0, size=N_OPP).astype(np.float32)
    mask, n_trades, assign, scores = build_mask(rng, edge, n_clusters, per, trade_mode)
    use_beta = beta_spread > 0
    sh1, sh2 = (0.0, 0.0) if use_beta else (s1, s2)
    train_lat = c.latent(a1, sh1, edge, rng)
    test_lat = c.latent(a2, sh2, edge, rng)
    beta = None
    if use_beta:
        beta = 1.0 + beta_spread * rng.uniform(-1.0, 1.0, size=n_cand)
    train_net, train_avg = outcomes(mask, train_lat, rng, s1, beta)
    mask_test = mask
    trades_test = None
    if test_trade_sigma > 0:
        n_test = np.clip(np.round(n_trades * np.exp(test_trade_sigma * rng.normal(0, 1, size=n_cand))), 100, 1000).astype(np.int32)
        mask_test = c.take(scores, n_test)
    test_net, test_avg = outcomes(mask_test, test_lat, rng, s2, beta)
    trades_test = np.maximum(mask_test.sum(axis=1), 1).astype(np.float64)
    return {
        "train_net": train_net,
        "train_avg": train_avg,
        "test_net": test_net,
        "test_avg": test_avg,
        "trades": np.maximum(mask.sum(axis=1), 1).astype(np.float64),
        "trades_test": trades_test,
        "assign": assign,
    }


def selftest() -> None:
    a = c.build_pair(1000, 1.00, c.SHIFT_STRONG, 0.80, c.SHIFT_WEAK)
    b = build_pair(1000, 1.00, c.SHIFT_STRONG, 0.80, c.SHIFT_WEAK)
    for key in ("train_net", "test_net", "trades"):
        assert np.array_equal(a[key], b[key]), key
    print("default worlds match spp_catalog_six")
    w = build_pair(1000, 1.00, c.SHIFT_STRONG, 0.80, c.SHIFT_WEAK, beta_spread=1.0, test_trade_sigma=0.3)
    print("beta + test trades:", round(float(w["test_net"].mean()), 1), round(float(np.corrcoef(w["trades"], w["trades_test"])[0, 1]), 2))
    w = build_pair(1000, 1.00, c.SHIFT_STRONG, 0.80, c.SHIFT_WEAK, trade_mode="candidate", n_clusters=10)
    print("trades per candidate, K=10:", round(float(w["trades"].std()), 1))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
