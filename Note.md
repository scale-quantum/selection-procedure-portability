# The sign of the selected candidate's OOS P&L mixes the period with the selection

Matvey Kiriakov · 11 October 2026

Walk-forward reports the sign of one number: the out-of-sample P&L of the candidate the rule ranked first in-sample. On a strong test period almost any candidate is in the black. On a weak one almost none is. In both cases the ordering among candidates may have survived, vanished, or flipped. The sign cannot tell these apart. The usual labels — non-robust, overfit, bad period — do not close the question.

**Selection Procedure Portability (SPP)** reads the same step through two things that are already on the table: the selected candidate's sum, and where the frozen top slice sits in the candidate set on the test. A comparison with candidates of the same trade count sits beside the sum. It does not replace the sum.

The paper, the NumPy library, and the testbed scripts are here:

[github.com/scale-quantum/selection-procedure-portability](https://github.com/scale-quantum/selection-procedure-portability)

## What the sign misses

I built a generator in which the period level and the ordering among candidates are set in advance (2,500 candidates, 50 clusters, 30 replicates per configuration). Five situations, same selection rule (top 10% by in-sample sum):

| Situation | Sign of selected "+" | Place of the top 10% |
| --- | --- | --- |
| Strong period, ordering preserved | 30 / 30 | 85 |
| Weak period, ordering preserved | 15 / 30 | 85 |
| Ordering flipped | 10 / 30 | 12 |
| No ordering, middle period | 14 / 30 | 48 |
| No ordering, strong period | 30 / 30 | 50 |

On a weak period with the ordering intact, the sign of the champion is a coin flip, while the frozen slice stays at the top of a losing set. On a strong period with no ordering, everyone is in the black, so the sign is always plus and cannot distinguish that from a real selection effect. The slice sits in the middle.

The share of pluses tracks how strong the test period is. That is a curve, not a frequency in the market:

![Share of replicates in which the reading matched the preset ordering, as a function of OOS period level.](figures/spp_level_curves.png)

## What this is not

It is not a replacement for PBO, White, or Hansen: those answer different questions. It is not a claim about how often the sign is wrong on other people's data. Six windows of one trading system and six windows of another algorithm show the same reading on real history. They are an illustration, not a rate.

Trade-count matching mutes, on this generator, the part of the period that enters P&L through how often a candidate trades. With a second level channel it can miss weak ordering on a weak segment. The matched row is a side diagnostic.

The library is `scripts/spp/`. It reads a table you already have. It does not download prices or run a backtest.

Paper and figures: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code: MIT.
