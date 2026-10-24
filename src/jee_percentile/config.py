"""Paths and modelling constants shared by every module."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("JEE_PCT_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACT_PATH = MODELS_DIR / "percentile_model_artifact.json"

# The model predicts the raw mark needed for each of these percentiles.
PERCENTILES = (90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0)

# Deployed feature set. Physics difficulty was the strongest, most stable
# signal; adding chemistry/maths only adds collinearity (see the ablation in
# reports/RESULTS.md). is_2026 anchors the year-level calibration and is fixed
# to 1 at inference time: a new paper is always scored against the latest year.
FEATURES = ("phy_hard", "phy_easy", "is_2026")
RIDGE_ALPHA = 10.0

# Central coverage of the prediction band reported alongside each estimate.
INTERVAL_COVERAGE = 0.80

SUBJECTS = {"physics": "phy", "chemistry": "chem", "mathematics": "math"}
