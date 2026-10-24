"""Leave-one-shift-out (LOSO) evaluation, baselines, ablations and uncertainty.

With only 29 labelled shifts, every number here comes from LOSO: each shift is
predicted by a model that never saw it. Two error scales are reported:

  percentile MAE  - place the true cutoff marks on the predicted curve and
                    read off the percentile; |that - true percentile|.
                    This is the error a student would actually experience.
  marks MAE       - |predicted cutoff marks - true cutoff marks|.

Confidence intervals are bootstrapped over shifts (the unit of independence).
"""

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from jee_percentile.config import (FEATURES, INTERVAL_COVERAGE, PERCENTILES, REPORTS_DIR,
                                   RIDGE_ALPHA)
from jee_percentile.curves import fit_all_shifts
from jee_percentile.data import load_curves, load_dataset, target_matrix
from jee_percentile.model import CutoffEnsemble, make_monotonic

P = np.asarray(PERCENTILES)

FEATURE_SETS = {
    "Year only": ("is_2026",),
    "Overall difficulty": ("overall_hard", "overall_easy", "is_2026"),
    "Physics difficulty (deployed)": FEATURES,
    "Physics + session time": FEATURES + ("is_evening",),
    "All three subjects": ("phy_hard", "phy_easy", "chem_hard", "chem_easy",
                           "math_hard", "math_easy", "is_2026"),
}


# ---------------------------------------------------------------- predictors

def _mean_curve(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    return np.tile(target_matrix(train).mean(axis=0), (len(test), 1))


def _year_mean_curve(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    out = []
    for _, row in test.iterrows():
        same = train[train["is_2026"] == row["is_2026"]]
        out.append(target_matrix(same if len(same) else train).mean(axis=0))
    return np.vstack(out)


def _ensemble(features, alpha=RIDGE_ALPHA):
    def predict(train, test):
        return CutoffEnsemble(features=tuple(features), alpha=alpha).fit(train).predict_curve(test)
    return predict


def loso_predictions(df: pd.DataFrame, predictor) -> np.ndarray:
    preds = np.zeros((len(df), len(P)))
    for i in range(len(df)):
        train = df.drop(index=df.index[i])
        preds[i] = make_monotonic(predictor(train, df.iloc[[i]])[0])
    return preds


# ------------------------------------------------------------------- metrics

def percentile_errors(actual: np.ndarray, pred: np.ndarray) -> np.ndarray:
    inferred = np.vstack([np.interp(a, p, P) for a, p in zip(actual, pred)])
    return np.abs(inferred - P)


def bootstrap_ci(per_shift: np.ndarray, n_boot: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_shift), size=(n_boot, len(per_shift)))
    means = per_shift[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@dataclass
class Score:
    name: str
    pct_mae: float
    pct_mae_ci: tuple
    pct_max: float
    marks_mae: float
    marks_rmse: float

    @classmethod
    def of(cls, name, actual, pred):
        pct_err = percentile_errors(actual, pred)
        marks_err = pred - actual
        return cls(name=name,
                   pct_mae=float(pct_err.mean()),
                   pct_mae_ci=bootstrap_ci(pct_err.mean(axis=1)),
                   pct_max=float(pct_err.max()),
                   marks_mae=float(np.abs(marks_err).mean()),
                   marks_rmse=float(np.sqrt((marks_err ** 2).mean())))


# --------------------------------------------------------------- uncertainty

def residual_quantiles(residuals: np.ndarray, coverage: float = INTERVAL_COVERAGE) -> dict:
    # finite-sample (split-conformal style) correction: with n residuals the
    # plain empirical quantile under-covers, so widen the level by (n+1)/n
    n = len(residuals)
    hi = min(1.0, (1 + coverage) / 2 * (n + 1) / n)
    lo = 1.0 - hi
    return {"coverage": coverage,
            "lower": np.quantile(residuals, lo, axis=0).round(3).tolist(),
            "upper": np.quantile(residuals, hi, axis=0).round(3).tolist()}


def interval_backtest(actual, pred, coverage=INTERVAL_COVERAGE) -> dict:
    """Honest coverage: each shift's band uses residual quantiles from the other shifts only."""
    residuals = actual - pred
    hits, widths_marks, widths_pct = [], [], []
    for i in range(len(actual)):
        q = residual_quantiles(np.delete(residuals, i, axis=0), coverage)
        lo, hi = pred[i] + np.asarray(q["lower"]), pred[i] + np.asarray(q["upper"])
        hits.append((actual[i] >= lo) & (actual[i] <= hi))
        widths_marks.append(hi - lo)
        lo_c, hi_c = make_monotonic(lo), make_monotonic(hi)
        widths_pct.append(np.interp(actual[i], lo_c, P) - np.interp(actual[i], hi_c, P))
    return {"nominal_coverage": coverage,
            "empirical_coverage": float(np.mean(hits)),
            "mean_width_marks": float(np.mean(widths_marks)),
            "mean_width_percentile_points": float(np.mean(widths_pct))}


# ------------------------------------------------------------------- reports

def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    feats = ["phy_hard", "phy_easy", "chem_hard", "chem_easy", "math_hard", "math_easy",
             "overall_hard", "overall_easy", "is_evening"]
    rows = []
    for f in feats:
        for p in (90.0, 95.0, 99.0):
            rho, pval = spearmanr(df[f], df[f"score_{p}"])
            rows.append({"feature": f, "target": f"marks@{p:g}", "rho": rho, "p_value": pval})
    return pd.DataFrame(rows)


def run(write: bool = True) -> dict:
    df = load_dataset()
    actual = target_matrix(df)

    preds = {
        "Mean curve (no features)": loso_predictions(df, _mean_curve),
        "Year-mean curve": loso_predictions(df, _year_mean_curve),
        "Cutoff ensemble (deployed)": loso_predictions(df, _ensemble(FEATURES)),
    }
    leaderboard = [Score.of(name, actual, pred) for name, pred in preds.items()]

    deployed = preds["Cutoff ensemble (deployed)"]
    per_shift = {name: percentile_errors(actual, pred).mean(axis=1) for name, pred in preds.items()}

    def improvement(baseline):
        gain = per_shift[baseline] - per_shift["Cutoff ensemble (deployed)"]
        return {"baseline": baseline, "mean": float(gain.mean()), "ci95": bootstrap_ci(gain),
                "relative": float(gain.mean() / per_shift[baseline].mean()),
                "shifts_improved": int((gain > 0).sum())}

    ablation = [Score.of(name, actual, loso_predictions(df, _ensemble(feats)))
                for name, feats in FEATURE_SETS.items()]
    alphas = [0.1, 0.3, 1, 3, 10, 30, 100]
    alpha_sweep = [{"alpha": a, "pct_mae": Score.of("", actual, loso_predictions(df, _ensemble(FEATURES, a))).pct_mae}
                   for a in alphas]

    per_percentile = percentile_errors(actual, deployed).mean(axis=0)
    stage1 = fit_all_shifts(load_curves())

    results = {
        "n_shifts": len(df),
        "n_shifts_with_imputed_tiers": int((df["n_imputed"] > 0).sum()),
        "leaderboard": [asdict(s) for s in leaderboard],
        "improvement": [improvement("Mean curve (no features)"), improvement("Year-mean curve")],
        "ablation": [asdict(s) for s in ablation],
        "alpha_sweep": alpha_sweep,
        "per_percentile_mae": dict(zip(map(str, PERCENTILES), per_percentile.round(4).tolist())),
        "prediction_interval": interval_backtest(actual, deployed),
        "residual_quantiles": residual_quantiles(actual - deployed),
        "stage1": {"n_shifts": len(stage1),
                   "rmse_pct_median": float(stage1["rmse_pct"].median()),
                   "rmse_pct_max": float(stage1["rmse_pct"].max()),
                   "by_year": stage1.groupby("year")["rmse_pct"].mean().round(3).to_dict()},
    }
    if write:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
        (REPORTS_DIR / "loso_predictions.csv").write_text(_predictions_csv(df, actual, deployed))
        spearman_table(df).round(4).to_csv(REPORTS_DIR / "spearman_correlations.csv", index=False)
        stage1.round(4).to_csv(REPORTS_DIR / "stage1_fits.csv", index=False)
        (REPORTS_DIR / "RESULTS.md").write_text(render_markdown(results, spearman_table(df)), encoding="utf-8")
    return results


def _predictions_csv(df, actual, pred) -> str:
    rows = []
    for sid, a, p in zip(df["shift_id"], actual, pred):
        for perc, av, pv in zip(PERCENTILES, a, p):
            rows.append({"shift_id": sid, "percentile": perc, "actual_marks": av, "predicted_marks": round(pv, 3)})
    return pd.DataFrame(rows).to_csv(index=False)


def _table(scores) -> str:
    lines = ["| Model | Percentile MAE | 95% CI | Max error | Marks MAE | Marks RMSE |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in scores:
        lo, hi = s["pct_mae_ci"]
        lines.append(f"| {s['name']} | **{s['pct_mae']:.3f}** | {lo:.3f} – {hi:.3f} | {s['pct_max']:.2f} "
                     f"| {s['marks_mae']:.2f} | {s['marks_rmse']:.2f} |")
    return "\n".join(lines)


def render_markdown(r: dict, spearman: pd.DataFrame) -> str:
    pi = r["prediction_interval"]
    gains = "\n".join(
        f"- vs **{g['baseline']}**: {g['mean']:.3f} percentile points lower error "
        f"({g['relative']:.0%}; 95% CI {g['ci95'][0]:.3f} – {g['ci95'][1]:.3f}), "
        f"better on {g['shifts_improved']} of {r['n_shifts']} shifts"
        for g in r["improvement"])
    top = spearman.reindex(spearman["rho"].abs().sort_values(ascending=False).index).head(8)
    corr = "\n".join(f"| `{t.feature}` | {t.target} | {t.rho:+.2f} | {t.p_value:.3f} |" for t in top.itertuples())
    pp = " | ".join(f"{float(k):g}" for k in r["per_percentile_mae"])
    pv = " | ".join(f"{v:.2f}" for v in r["per_percentile_mae"].values())
    return f"""# Results

_Auto-generated by `python -m jee_percentile evaluate` - do not edit by hand._

Leave-one-shift-out cross-validation over **{r['n_shifts']} JEE Main shifts**
({r['n_shifts_with_imputed_tiers']} with 90/91 tiers imputed from their own Stage-1 fit).

## Leaderboard

{_table(r['leaderboard'])}

Paired bootstrap over shifts:

{gains}

## Feature ablation

{_table(r['ablation'])}

## Error by percentile tier

| Percentile | {pp} |
|---|{'---:|' * len(r['per_percentile_mae'])}
| MAE (pts) | {pv} |

## Prediction interval

Nominal {pi['nominal_coverage']:.0%} band, back-tested on held-out shifts:
empirical coverage **{pi['empirical_coverage']:.1%}**, mean width {pi['mean_width_marks']:.1f} marks
(≈ {pi['mean_width_percentile_points']:.2f} percentile points).

## Strongest univariate signals (Spearman)

| Feature | Target | ρ | p-value |
|---|---|---:|---:|
{corr}

## Stage 1 curve fits

{r['stage1']['n_shifts']} shifts fit with a Normal CDF; median RMSE {r['stage1']['rmse_pct_median']:.3f}
percentile points (worst {r['stage1']['rmse_pct_max']:.3f}).
"""


if __name__ == "__main__":
    out = run()
    for row in out["leaderboard"]:
        print(f"{row['name']:32s} pct MAE {row['pct_mae']:.4f}  marks MAE {row['marks_mae']:.2f}")
