# Selection Procedure Portability

Matvey Kiriakov

Selection Procedure Portability (SPP) reads one out-of-sample step of a selection rule: the sum of the candidate that was frozen in-sample, and where that slice sits in the candidate set. A comparison with candidates of the same trade count sits beside the sum. It does not replace the sum.

The paper is [Selection Procedure Portability](Selection%20Procedure%20Portability.md).

## Read your own table

The library is `scripts/spp/` (NumPy only). It does not download prices or run a backtest. From `scripts/`:

```python
from spp import gap_matched, selection_report, top_index

chosen = top_index(train_score, 0.10)
report = selection_report(train_score, test_values, fraction=0.10)
matched = gap_matched(test_values, chosen, trade_count)

print(report["champion_value"], report["champion_place"])
print(report["place"], report["gap_exact"])
print(matched["gap_matched"])
```

`train_score`, `test_values` and `trade_count` are columns of the same length and the same row order. Details are in [scripts/spp/README.md](scripts/spp/README.md).

## Reproduce the testbed

From the repository root, run the commands in Appendix C of the paper, except `spp_live_table.py` and `spp_build_article.py`. Those two are not in this repository: the historical matrices are not published. The studies need NumPy. Regenerating the figures also needs matplotlib, after `spp_baseline_study.py` has written its results.

## License

The paper and figures are [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): reuse with attribution. The code in `scripts/` is [MIT](LICENSE).
