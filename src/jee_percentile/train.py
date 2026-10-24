"""Fit the deployed model on every shift and export the JSON artifact."""

import json

from jee_percentile.config import ARTIFACT_PATH, FEATURES
from jee_percentile.data import load_dataset, target_matrix
from jee_percentile.evaluate import _ensemble, loso_predictions, residual_quantiles
from jee_percentile.model import CutoffEnsemble


def train() -> CutoffEnsemble:
    df = load_dataset()
    model = CutoffEnsemble().fit(df)
    oos = loso_predictions(df, _ensemble(FEATURES))
    model.residual_quantiles = residual_quantiles(target_matrix(df) - oos)
    return model


def export(path=ARTIFACT_PATH) -> dict:
    artifact = train().to_artifact()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


if __name__ == "__main__":
    a = export()
    print(f"wrote {ARTIFACT_PATH} ({a['n_shifts_trained_on']} shifts, features {a['features']})")
