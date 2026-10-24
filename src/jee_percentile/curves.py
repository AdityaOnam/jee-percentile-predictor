"""Stage 1: compress each shift's percentile table into a Normal-CDF fingerprint.

percentile = 100 * Phi((marks - mu) / sigma)

The fit is monotonic by construction and summarises a 10-23 point table with
two numbers. Published tables only cover the upper tail (90th+ percentile) of a
right-skewed distribution, so mu is often negative: it is a location parameter
of the tail fit, not the mean score.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm


def normal_cdf_pct(marks, mu, sigma):
    return 100.0 * norm.cdf((marks - mu) / sigma)


def normal_cdf_inverse(percentile, mu, sigma):
    return mu + sigma * norm.ppf(percentile / 100.0)


def fit_normal_cdf(marks: np.ndarray, percentiles: np.ndarray):
    """Least-squares Normal CDF fit; returns None if it fails to converge."""
    marks = np.asarray(marks, dtype=float)
    percentiles = np.asarray(percentiles, dtype=float)
    p0 = [marks.mean(), marks.std() if marks.std() > 0 else 20.0]
    try:
        (mu, sigma), _ = curve_fit(normal_cdf_pct, marks, percentiles, p0=p0, maxfev=10000)
    except RuntimeError:
        return None
    residuals = normal_cdf_pct(marks, mu, sigma) - percentiles
    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "rmse_pct": float(np.sqrt(np.mean(residuals ** 2))),
        "n_points": int(len(marks)),
    }


def fit_all_shifts(curves: pd.DataFrame) -> pd.DataFrame:
    """Stage-1 fingerprint for every shift in a long-format curves table."""
    rows = []
    for sid, grp in curves.groupby("shift_id", sort=True):
        fit = fit_normal_cdf(grp["raw_score"].to_numpy(), grp["percentile"].to_numpy())
        if fit is not None:
            rows.append({"shift_id": sid, "year": grp["year"].iat[0], "session": grp["session"].iat[0], **fit})
    return pd.DataFrame(rows)
