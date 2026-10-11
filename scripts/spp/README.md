# SPP

Selection Procedure Portability. The sign of one champion mixes how the next
window treated everyone with whether the slice you froze kept its place.
This package reads a table you already computed. It does not download prices
or run a backtest.

Run Python from `scripts/` so `import spp` works. From the repository root the
import fails unless `scripts/` is on `sys.path`. NumPy only.

## Read one step

Rows are candidates. Every column has the same length and the same row order.
`train_score` is the number you ranked on (net, average trade, drawdown).
`test_values` is what those same rows did next, usually net. `trade_count` is
how many trades each row took on the training window. The chosen rows are not
re-picked on the test column.

```python
from spp import decile_table, gap_matched, selection_report, top_index, top_overlap

chosen = top_index(train_score, 0.10)
report = selection_report(train_score, test_values, fraction=0.10)
matched = gap_matched(test_values, chosen, trade_count)

print(report["champion_value"], report["champion_place"])  # one frozen name
print(report["place"], report["selected_mean"], report["gap_exact"])  # the slice
print(matched["gap_matched"])  # same trade-count, beside the sum
print(decile_table(train_score, test_values))  # decile 1 is the top of train_score
print(top_overlap(train_score, test_values, ks=(100,)))
```

The reading of the step is the first two `print` lines. `champion_value` and `champion_place` are the
single best training row, read on the test column. Place 0 is worst, just under
100 is best. `place` is the median place of the frozen slice, not the place of
its average. `gap_exact` is `selected_mean` minus the mean of the whole table:
the expectation of a same-sized draw, with no random draws. That is the gap the
article reports.

`gap_matched` is the side line. It does not replace the sum. Rows go into ten
quantile bins of `trade_count`. The function subtracts the exact expectation of
a draw that takes from each bin as many rows as the slice does. No random draws.
If every row has the same trade count, `gap_matched` equals `gap_plain`, and
`gap_plain` equals `gap_exact`. For a rule that selects on trade count itself,
the matched gap is zero by construction, up to ties. On the synthetic worlds
where the window level reaches the sum only through trade count, the plain gap
moves with that level and the matched gap stays near zero. When the level has
another channel, the matched gap does not remove it. There is no transfer
threshold in this package.

`gap`, `z` and `random_*` are also in the report. They come from `draws` random
sets (default 300). If you omit `rng`, those draws use seed 0, so a second call
returns the same `gap` and `z`. Pass your own `numpy.random.Generator` when you
want a new draw. The article does not use `z` and does not report `gap`.

The dictionary includes:

| Field | Meaning |
| --- | --- |
| `champion_value`, `champion_place` | The single best training row, read on the test column. Place 0 is worst, just under 100 is best. |
| `place` | Median place of the frozen slice, not the place of its average. |
| `selected_mean`, `selected_positive` | Mean of the slice, and the share of its rows above zero. |
| `universe_mean`, `universe_positive` | The same for every row. |
| `random_mean`, `random_positive` | Same-sized draws without replacement. |
| `random_sd`, `random_p05`, `random_p50`, `random_p95` | Spread of those draw means. |
| `gap` | `selected_mean` minus `random_mean`. |
| `gap_exact` | `selected_mean` minus `universe_mean`. No draws. |
| `z` | Gap divided by `random_sd`. Not a test and not a p-value. Empty when the slice is the whole table. |
| `random_ge_selected` | Share of draws whose mean was at least the selected mean. `0.0` means none of `draws`. |

## Several fractions and several rules

This is the live-fold table: 1%, 5%, 10%, 30%, then more than one training column
against the same test column.

```python
from spp import compare_rules, selection_profile

by_net = selection_profile(train_score, test_values, draws=10_000)
both = compare_rules(
    {"net": train_score, "average_trade": average_trade},
    test_values,
    draws=10_000,
)
```

`by_net` is a list of the same dictionaries, one per fraction.
Each row of `both` adds `rule`.

Rank correlation of two columns of the **same** metric is `spearman(train_score, test_values)` (`from spp import spearman`).
Do not read a correlation of average-trade ranks against net as if it were
the net-to-net number from the article.

`places_matched(test_values, trade_count)` is the rank counterpart of the matched
gap: the place of each row among rows with a similar trade count. A slice's
matched place is the median of these. `spearman_matched(train_score, test_values, trade_count)`
is Spearman inside the same ten bins, averaged with bin size as the weight.
On the stand, a plain Spearman over the whole table moves with the window level
when the sum depends on trade count; the matched one does not (article, 4.11).

`top_overlap` reports how many names sit in the top k on both windows, and
`chance_share` = k / n, the share a blind top k would share. The matched-gap
numbers in the article are reproduced by `scripts/spp_worlds_study.py`
(section 4.1) and the level curves in `scripts/spp_shift_study.py`.

## What this will not accept

Different lengths, NaN, a fraction outside (0, 1], a true/false mask where row
numbers belong, and fewer than two random draws. Those raise `ValueError`.
If every value in a column is the same, `spearman` is undefined (`nan`), not 1.

Ties keep their row order. Two identical sums do not get averaged places.
The single best row (`champion_value`, `champion_place`) is the last of the tied
rows, the same row `top_index` puts in a one-row slice.
Shuffling rows can change who sits on the cut of a fraction. That is the same
rule the article's generator uses.

## Reproduce the testbed

The paper is `Selection Procedure Portability.md` at the repository root.
Run from the repository root:

```powershell
python scripts/spp_catalog_six.py
python scripts/spp_worlds_study.py
python scripts/spp_shift_study.py
python scripts/spp_ribbon_study.py
python scripts/spp_ladder_study.py
python scripts/spp_cluster_study.py
python scripts/spp_baseline_study.py
python scripts/spp_robust_study.py
python scripts/spp_cluster_decoupled.py
python scripts/spp_clusternull_study.py
python scripts/spp_clusternull_study.py --mode candidate
python scripts/spp_plot_article.py
```

Each study writes its report and data next to itself. The worlds study takes
about 3 minutes, the shift study about 1 minute, the chain study about 10
seconds, the catalog about 30 seconds. Runs are deterministic
(`numpy.random.default_rng`). To import `spp` from your own code, run from
`scripts/` or put that folder on `sys.path`.

| What | Seeds |
| --- | --- |
| Configurations | 1000–1029 (configuration 5 shifted by 5000) |
| Null worlds, thresholds | 2000–2199 |
| Held-out null worlds | 8000–8199 |
| Order sweep | 3000–3029 |
| Chains | 4000–4029 |
| Level curves | 20000–20029 (medium-train type shifted by 5000) |
| Illustrative seed | 41 |

Seeds 41, 100, 7 and 19 were used while choosing shifts. No frequency in the
paper uses them, except the seed 41 illustration. The historical windows in
section 5 are not in this repository, so they cannot be recomputed from here.
The library itself needs only NumPy. Regenerating the figures also needs
matplotlib, after the baseline study has written its results.
