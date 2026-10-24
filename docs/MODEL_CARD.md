# Model card: JEE Main cutoff ensemble v1.1.0

## Intended use

Estimate a JEE Main percentile (90–99 range) from a raw score out of 300 and the Physics difficulty
mix of the paper the student sat. Typical uses:

- a fresh shift on exam day, before official percentile tables are published
- a full-length practice test whose difficulty profile is known

**Not for:** admission decisions, rank guarantees, or scores outside the 90–99 percentile range.

## Inputs

| Feature | Meaning | Training range |
|---|---|---|
| `phy_hard` | % of Physics questions labelled Hard | 0 – 12 |
| `phy_easy` | % of Physics questions labelled Easy | 20 – 64 |
| `is_2026` | year anchor; always 1 at inference (scored against the latest year) | 0 / 1 |

## Architecture

1. Ten ridge regressions (α = 10, standardised features), one per percentile 90…99, each predicting the raw mark needed for that percentile.
2. Isotonic regression (PAVA) enforces a non-decreasing cutoff curve.
3. Linear interpolation of the student's marks onto the curve.
4. An 80% band from per-percentile quantiles of leave-one-shift-out residuals, with a finite-sample `(n+1)/n` level correction.

## Training data

29 JEE Main shifts (January 2025, April 2025, January 2026) that have both a published marks-vs-percentile
table and per-question difficulty labels. The 90th and 91st tiers of the 9 April 2025 shifts are imputed
from each shift's own Normal-CDF fit. See [`data/README.md`](../data/README.md).

## Evaluation (leave-one-shift-out)

| Metric | Value |
|---|---|
| Percentile MAE | 0.716 (95% CI 0.52 – 0.93) |
| Max percentile error | 3.51 |
| Marks MAE / RMSE | 6.27 / 8.36 |
| 99th-tier percentile MAE | 0.16 |
| 80% band, empirical coverage | 80.0% |
| vs mean curve | −46% error (CI excludes 0) |
| vs year-mean curve | −15% error (CI includes 0) |

Full tables: [`reports/RESULTS.md`](../reports/RESULTS.md).

## Known limitations and risks

- Small sample (n = 29). Coefficients are directional, not precise.
- Difficulty labels come from one annotator's question bank. 2026 subject labels contain 14 known misfilings.
- Percentile tables come from coaching institutes and aggregators, not NTA. Only January 2025 is cross-validated across two sources.
- A paper outside the training difficulty range is an extrapolation. The band does not account for that.
