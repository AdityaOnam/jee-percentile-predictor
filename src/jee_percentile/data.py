"""Load the curated JSON datasets into tidy DataFrames.

Three views of the same exam shifts:
  load_curves()    every shift's published (marks -> percentile) table, long format
  load_features()  per-shift difficulty composition (percent easy / hard per subject)
  load_targets()   per-shift raw mark needed for each percentile in config.PERCENTILES
  load_dataset()   features joined with targets - the modelling table
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from jee_percentile.config import DATA_DIR, PERCENTILES, SUBJECTS
from jee_percentile.curves import fit_normal_cdf, normal_cdf_inverse


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _session(shift_id: str) -> str:
    month = shift_id.split("_")[2]
    return "January" if month in {"JAN", "FEB"} else "April"


def load_curves(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per (shift, percentile tier) from every shift-wise percentile file."""
    rows = []
    for path in sorted((data_dir / "percentiles").glob("*-percentiles.json")):
        for shift in _read(path).get("shifts", []):
            for point in shift.get("percentile_curve", []):
                rows.append({
                    "shift_id": shift["shift_id"],
                    "year": int(shift["shift_id"][:4]),
                    "session": _session(shift["shift_id"]),
                    "date": shift.get("date"),
                    "percentile": float(point["percentile"]),
                    "raw_score": float(point["raw_score"]),
                })
    return pd.DataFrame(rows)


def load_features(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Per-shift difficulty composition, in percent of questions."""
    rows = []
    for path in sorted((data_dir / "difficulty_subject").glob("*.json")):
        for shift in _read(path).get("shifts", []):
            sid = shift["shift_id"]
            row = {
                "shift_id": sid,
                "year": int(sid[:4]),
                "session": _session(sid),
                "date": shift.get("date"),
                "is_2026": int(sid.startswith("2026")),
                "is_evening": int(str(shift.get("session_time", "")).lower() == "evening"),
            }
            totals = {"total": 0, "n_easy": 0, "n_hard": 0}
            for subject, prefix in SUBJECTS.items():
                counts = shift["subjects"].get(subject, {"total": 1, "n_easy": 0, "n_hard": 0})
                row[f"{prefix}_hard"] = 100.0 * counts["n_hard"] / counts["total"]
                row[f"{prefix}_easy"] = 100.0 * counts["n_easy"] / counts["total"]
                for key in totals:
                    totals[key] += counts[key]
            row["overall_hard"] = 100.0 * totals["n_hard"] / totals["total"]
            row["overall_easy"] = 100.0 * totals["n_easy"] / totals["total"]
            rows.append(row)
    return pd.DataFrame(rows)


def load_targets(data_dir: Path = DATA_DIR, extrapolate_missing: bool = True) -> pd.DataFrame:
    """Raw mark needed for each modelled percentile, one row per shift.

    April 2025 tables start at the 92nd percentile. When a shift has at least
    six published tiers, its curve is fit with a Normal CDF and inverted to
    fill the missing 90/91 tiers - a small, bounded extrapolation that keeps
    9 extra shifts in the training set.
    """
    curves = load_curves(data_dir)
    rows = []
    for sid, grp in curves.groupby("shift_id", sort=True):
        known = dict(zip(grp["percentile"], grp["raw_score"]))
        missing = [p for p in PERCENTILES if p not in known]
        if missing and extrapolate_missing and len(known) >= 6:
            fit = fit_normal_cdf(grp["raw_score"].to_numpy(), grp["percentile"].to_numpy())
            if fit is not None:
                for p in missing:
                    known[p] = float(normal_cdf_inverse(p, fit["mu"], fit["sigma"]))
                missing = []
        if missing:
            continue
        row = {"shift_id": sid, "n_imputed": sum(p not in set(grp["percentile"]) for p in PERCENTILES)}
        row.update({f"score_{p}": known[p] for p in PERCENTILES})
        rows.append(row)
    return pd.DataFrame(rows)


def target_columns() -> list:
    return [f"score_{p}" for p in PERCENTILES]


def load_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Modelling table: one row per shift that has both difficulty data and a percentile curve."""
    df = load_features(data_dir).merge(load_targets(data_dir), on="shift_id", how="inner")
    return df.sort_values("shift_id").reset_index(drop=True)


def target_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[target_columns()].to_numpy(dtype=float)
