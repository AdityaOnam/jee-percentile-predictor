"""Dependency-free inference from the exported JSON artifact.

Standard library only (no numpy / scikit-learn / scipy), so this file can be
dropped into any backend service as-is. tests/test_portable.py checks it
against the scikit-learn model to 1e-9.
"""

import json
from pathlib import Path


def load_artifact(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pava(values):
    """Pool Adjacent Violators: least-squares non-decreasing fit, i.e. sklearn's
    IsotonicRegression(increasing=True) on equally weighted points."""
    blocks = []  # [sum, count]
    for v in values:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out = []
    for s, c in blocks:
        out.extend([s / c] * c)
    return out


def interp(x, xp, fp):
    """numpy.interp for non-decreasing xp, clipped at both ends."""
    if x <= xp[0]:
        return fp[0]
    if x >= xp[-1]:
        return fp[-1]
    for i in range(1, len(xp)):
        if x <= xp[i]:
            if xp[i] == xp[i - 1]:
                return fp[i - 1]
            t = (x - xp[i - 1]) / (xp[i] - xp[i - 1])
            return fp[i - 1] + t * (fp[i] - fp[i - 1])
    return fp[-1]


def predict_curve(artifact, phy_hard_pct, phy_easy_pct):
    """(percentiles, monotonic cutoff marks) for a paper with this Physics difficulty."""
    raw = {"phy_hard": phy_hard_pct, "phy_easy": phy_easy_pct, "is_2026": 1}
    mean, scale = artifact["scaler"]["mean"], artifact["scaler"]["scale"]
    z = {f: (raw[f] - mean[f]) / scale[f] for f in artifact["features"]}
    percentiles = sorted(float(p) for p in artifact["percentile_cutoffs"])
    marks = []
    for p in percentiles:
        cut = artifact["percentile_cutoffs"][str(p)]
        marks.append(cut["intercept"] + sum(cut["coefficients"][f] * z[f] for f in z))
    return percentiles, pava(marks)


def predict_percentile(artifact, marks, phy_hard_pct, phy_easy_pct):
    """Returns (low, point, high) percentile; low == high when no interval is stored."""
    percentiles, curve = predict_curve(artifact, phy_hard_pct, phy_easy_pct)
    point = interp(marks, curve, percentiles)
    band = artifact.get("prediction_interval")
    if not band:
        return point, point, point
    hi_curve = pava([c + u for c, u in zip(curve, band["upper"])])
    lo_curve = pava([c + l for c, l in zip(curve, band["lower"])])
    return interp(marks, hi_curve, percentiles), point, interp(marks, lo_curve, percentiles)


if __name__ == "__main__":
    art = load_artifact(Path(__file__).resolve().parents[2] / "models" / "percentile_model_artifact.json")
    low, pct, high = predict_percentile(art, marks=180, phy_hard_pct=10, phy_easy_pct=20)
    print(f"180 marks, physics 10% hard / 20% easy -> {pct:.2f} percentile (80% band {low:.2f} - {high:.2f})")
