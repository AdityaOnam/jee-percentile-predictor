"""Stage 2: an ensemble of per-percentile ridge regressions.

For every percentile p in config.PERCENTILES a separate ridge regression maps a
shift's difficulty features to the raw mark needed to reach p. Independently
fit cutoffs carry no guarantee of being ordered, so the predicted curve is
passed through isotonic regression before a student's marks are interpolated
onto it. Isotonic regression keeps each value attached to its own percentile,
unlike sorting, which would silently swap them if two cutoffs crossed.

Uncertainty: the model stores the empirical quantiles of its out-of-sample
cutoff residuals (from leave-one-shift-out CV). Shifting the predicted curve by
those quantiles gives a band of plausible curves, and therefore a percentile
range for any mark.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from jee_percentile.config import FEATURES, PERCENTILES, RIDGE_ALPHA
from jee_percentile.data import target_matrix


def make_monotonic(curve) -> np.ndarray:
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    return iso.fit_transform(np.asarray(PERCENTILES), np.asarray(curve, dtype=float))


@dataclass
class CutoffEnsemble:
    features: tuple = FEATURES
    alpha: float = RIDGE_ALPHA
    residual_quantiles: dict = field(default_factory=dict)  # {"lower": [...], "upper": [...], "coverage": c}

    def fit(self, df: pd.DataFrame) -> "CutoffEnsemble":
        X = df[list(self.features)].to_numpy(dtype=float)
        Y = target_matrix(df)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.models_ = [Ridge(alpha=self.alpha).fit(Xs, Y[:, j]) for j in range(Y.shape[1])]
        self.n_train_ = len(df)
        return self

    def _matrix(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X[list(self.features)].to_numpy(dtype=float)
        if isinstance(X, dict):
            return np.array([[X[f] for f in self.features]], dtype=float)
        return np.atleast_2d(np.asarray(X, dtype=float))

    def predict_raw(self, X) -> np.ndarray:
        """Unconstrained cutoff marks, shape (n_rows, n_percentiles)."""
        Xs = self.scaler_.transform(self._matrix(X))
        return np.column_stack([m.predict(Xs) for m in self.models_])

    def predict_curve(self, X) -> np.ndarray:
        """Monotonic cutoff marks, shape (n_rows, n_percentiles)."""
        return np.vstack([make_monotonic(row) for row in self.predict_raw(X)])

    def predict_percentile(self, marks: float, X) -> float:
        curve = self.predict_curve(X)[0]
        return float(np.interp(marks, curve, PERCENTILES))

    def predict_interval(self, marks: float, X):
        """(low, point, high) percentile estimate for one paper."""
        curve = self.predict_curve(X)[0]
        point = float(np.interp(marks, curve, PERCENTILES))
        if not self.residual_quantiles:
            return point, point, point
        # a harder-than-predicted curve (cutoffs higher) means a lower percentile
        hi_curve = make_monotonic(curve + np.asarray(self.residual_quantiles["upper"]))
        lo_curve = make_monotonic(curve + np.asarray(self.residual_quantiles["lower"]))
        low = float(np.interp(marks, hi_curve, PERCENTILES))
        high = float(np.interp(marks, lo_curve, PERCENTILES))
        return low, point, high

    @staticmethod
    def query(phy_hard: float, phy_easy: float) -> dict:
        """Feature row for a new paper; always anchored to the latest reference year."""
        return {"phy_hard": phy_hard, "phy_easy": phy_easy, "is_2026": 1}

    def to_artifact(self) -> dict:
        """Coefficients in a dependency-free JSON form (see jee_percentile.portable)."""
        return {
            "model_version": "1.1.0",
            "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_shifts_trained_on": self.n_train_,
            "features": list(self.features),
            "ridge_alpha": self.alpha,
            "scaler": {
                "mean": dict(zip(self.features, map(float, self.scaler_.mean_))),
                "scale": dict(zip(self.features, map(float, self.scaler_.scale_))),
            },
            "percentile_cutoffs": {
                str(p): {"intercept": float(m.intercept_),
                         "coefficients": dict(zip(self.features, map(float, m.coef_)))}
                for p, m in zip(PERCENTILES, self.models_)
            },
            "prediction_interval": self.residual_quantiles or None,
            "inference_notes": "is_2026 is always 1 at inference. Cutoffs are made monotonic "
                               "with isotonic regression (PAVA) before interpolating marks.",
        }
