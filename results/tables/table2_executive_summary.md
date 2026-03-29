# Table 2: Cross-Metric Robustness Summary

This table is intentionally aggregated (mean ± std across selected severities), so it complements the CSV instead of duplicating row-level metrics.

| Group | Mean ΔBLEU (pp) | Mean ΔWER (pp) | Mean ΔROUGE (pp)  | Worst cond (ΔBLEU) |
|-------|----------------:|---------------:|------------------:|--------------------|
| HT | 3.51 ± 2.65 | 8.10 ± 5.96 | 4.30 ± 3.19 | HT_20 (7.06) |
| TT | 1.90 ± 1.69 | 6.22 ± 4.96 | 1.89 ± 1.50 | TT_20 (4.24) |
| HC | 3.38 ± 2.11 | 7.58 ± 5.54 | 2.81 ± 1.94 | HC_20 (6.21) |
| TC | 2.93 ± 2.11 | 8.60 ± 7.03 | 2.37 ± 1.78 | TC_20 (5.75) |
| HT+TT | 5.67 ± 3.29 | 14.27 ± 8.02 | 6.20 ± 3.71 | HT_20+TT_20 (12.02) |
| HT+TC | 5.92 ± 2.76 | 16.78 ± 9.42 | 6.07 ± 3.20 | HT_20+TC_20 (11.12) |
| TT+HC | 5.13 ± 2.42 | 13.59 ± 7.67 | 4.75 ± 2.46 | HC_20+TT_20 (9.18) |
| HC+TC | 6.11 ± 2.49 | 15.69 ± 8.71 | 4.88 ± 2.37 | HC_20+TC_20 (10.35) |
| All (basic+compound) | 5.01 ± 2.92 | 13.22 ± 8.63 | 4.82 ± 3.12 | HT_20+TT_20 (12.02) |