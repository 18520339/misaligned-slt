# Table 2: Cross-Metric Robustness Summary

This table is intentionally aggregated (mean ± std across selected severities), so it complements the CSV instead of duplicating row-level metrics.

| Group | Mean ΔBLEU (pp) | Mean ΔWER (pp) | Mean ΔROUGE (pp)  | Worst cond (ΔBLEU) |
|-------|----------------:|---------------:|------------------:|--------------------|
| HT | 3.60 ± 2.58 | 8.07 ± 5.96 | 4.41 ± 3.13 | HT_20 (7.05) |
| TT | 2.07 ± 1.73 | 6.15 ± 4.96 | 2.03 ± 1.54 | TT_20 (4.47) |
| HC | 3.44 ± 2.10 | 7.39 ± 5.52 | 2.86 ± 1.87 | HC_20 (6.22) |
| TC | 2.93 ± 2.06 | 8.33 ± 7.09 | 2.37 ± 1.80 | TC_20 (5.66) |
| HT+TT | 5.76 ± 3.29 | 14.19 ± 8.03 | 6.31 ± 3.68 | HT_20+TT_20 (12.06) |
| HT+TC | 5.94 ± 2.73 | 16.61 ± 9.46 | 6.07 ± 3.24 | HT_20+TC_20 (11.02) |
| TT+HC | 5.30 ± 2.48 | 13.46 ± 7.66 | 4.86 ± 2.49 | HC_20+TT_20 (9.48) |
| HC+TC | 6.11 ± 2.54 | 15.42 ± 8.70 | 4.87 ± 2.39 | HC_20+TC_20 (10.43) |
| All (basic+compound) | 5.09 ± 2.92 | 13.06 ± 8.63 | 4.87 ± 3.12 | HT_20+TT_20 (12.06) |