import json

import numpy as np
import pytest

from jee_percentile import portable
from jee_percentile.config import PERCENTILES
from jee_percentile.curves import fit_normal_cdf, normal_cdf_inverse, normal_cdf_pct
from jee_percentile.data import load_dataset, target_matrix
from jee_percentile.model import CutoffEnsemble, make_monotonic
from jee_percentile.train import train


@pytest.fixture(scope="module")
def df():
    return load_dataset()


@pytest.fixture(scope="module")
def model():
    return train()


def test_dataset_shape(df):
    assert len(df) == 29
    assert not df[[f"score_{p}" for p in PERCENTILES]].isna().any().any()
    assert set(df["year"]) == {2025, 2026}


def test_published_cutoffs_increase_with_percentile(df):
    assert (np.diff(target_matrix(df), axis=1) >= 0).all()


def test_normal_cdf_roundtrip():
    marks = np.array([80, 100, 120, 150, 180])
    pct = normal_cdf_pct(marks, -10, 70)
    fit = fit_normal_cdf(marks, pct)
    assert fit["rmse_pct"] < 1e-6
    assert normal_cdf_inverse(95.0, fit["mu"], fit["sigma"]) == pytest.approx(-10 + 70 * 1.6448536, rel=1e-4)


def test_curve_is_monotonic_over_feature_space(model):
    for hard in range(0, 21, 2):
        for easy in range(10, 71, 5):
            assert (np.diff(model.predict_curve(model.query(hard, easy))[0]) >= 0).all()


def test_harder_paper_needs_fewer_marks(model):
    easy_paper = model.predict_curve(model.query(0, 56))[0]
    hard_paper = model.predict_curve(model.query(12, 28))[0]
    assert (hard_paper < easy_paper).all()


def test_interval_contains_point(model):
    low, point, high = model.predict_interval(120, model.query(4, 40))
    assert low <= point <= high
    assert 90 <= low and high <= 99


def test_isotonic_fixes_crossing():
    curve = [100, 99, 101, 110, 108, 120, 130, 140, 150, 160]
    fixed = make_monotonic(curve)
    assert (np.diff(fixed) >= 0).all()
    assert portable.pava(curve) == pytest.approx(list(fixed))


@pytest.mark.parametrize("marks,hard,easy", [(75, 0, 56), (120, 4, 40), (150, 12, 28), (60, 8, 20), (250, 2, 50)])
def test_portable_matches_sklearn(model, marks, hard, easy):
    artifact = json.loads(json.dumps(model.to_artifact()))
    p_low, p_pt, p_high = portable.predict_percentile(artifact, marks, hard, easy)
    s_low, s_pt, s_high = model.predict_interval(marks, model.query(hard, easy))
    assert p_pt == pytest.approx(s_pt, abs=1e-9)
    assert p_low == pytest.approx(s_low, abs=1e-9)
    assert p_high == pytest.approx(s_high, abs=1e-9)


def test_reproduces_reported_loso_mae(df):
    from jee_percentile.evaluate import _ensemble, loso_predictions, percentile_errors
    pred = loso_predictions(df, _ensemble(CutoffEnsemble().features))
    assert percentile_errors(target_matrix(df), pred).mean() == pytest.approx(0.7161, abs=1e-3)
