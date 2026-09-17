"""
HealthConnect Clinic -- No-Show Prediction
Model-integration interface (Week 7 refinement)

This replaces the Week 6 `NoShowModelInterface`. The Week 6 version hardcoded
a single global feature set (FEATURE_COLS / engineer_features), which meant it
could only host models trained on exactly that feature set. Week 7 testing
proved this concretely: wrapping a model that needed one extra feature
(booking_month, from the Data Science track's spec) failed outright with
"columns are missing: {'booking_month'}".

NoShowModelInterfaceV2 fixes this by taking the feature-engineering function
and the feature list as constructor arguments, per model, instead of reading
them from module-level globals. It also adds two things the Week 6 version
was missing:

  1. Numeric dtype validation before scoring, so a malformed value (e.g. a
     text string in a numeric column) raises a clear, named error instead of
     a raw exception several layers deep inside scikit-learn.
  2. Logging on every scoring call, instead of silent execution.

Usage
-----
    from src.features import engineer_features, FEATURE_COLS, NUMERIC_FEATURES
    from src.models.interface import NoShowModelInterfaceV2

    baseline_wrapped = NoShowModelInterfaceV2(
        fitted_pipeline=model,
        name="LogisticRegression (Week 5 baseline)",
        source="ML Engineering",
        feature_fn=engineer_features,
        feature_cols=FEATURE_COLS,
        numeric_cols=NUMERIC_FEATURES,
    )
    risk_scores = baseline_wrapped.predict_risk(raw_appointments_df)

A model with a different feature set (e.g. one that adds booking_month) is
wrapped the same way, just with its own feature_fn/feature_cols/numeric_cols:

    data_science_wrapped = NoShowModelInterfaceV2(
        fitted_pipeline=model_with_booking_month,
        name="LogisticRegression + booking_month (Data Science spec)",
        source="Data Science",
        feature_fn=engineer_features_v2,
        feature_cols=FEATURE_COLS_V2,
        numeric_cols=NUMERIC_FEATURES,
    )
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import pandas as pd

logger = logging.getLogger("healthconnect.pipeline")

# Raw columns every caller must provide, regardless of which model/feature
# set is behind the interface. These are needed by feature engineering
# itself (e.g. computing prior_no_show_rate), not just by the model.
REQUIRED_RAW_COLUMNS = {
    "appointment_id",
    "previous_appointments",
    "previous_no_shows",
    "booking_lead_days",
    "reminder_sent",
    "reminder_channel",
}


class NoShowModelInterfaceV2:
    """
    A documented, tested contract for scoring HealthConnect appointments
    with any fitted model -- this track's own models, or a model handed
    over by the Data Science track -- without touching ingestion or
    feature-engineering code when the underlying model changes.

    Parameters
    ----------
    fitted_pipeline : object
        A fitted scikit-learn-compatible pipeline exposing
        ``predict_proba(X)``. Expected to accept exactly the columns
        in ``feature_cols``, in any order, as a DataFrame.
    name : str
        Human-readable model name, used in log messages and error text
        (e.g. "LogisticRegression (Week 5 baseline)").
    source : str
        Which track/person this model came from (e.g. "ML Engineering",
        "Data Science"). Purely descriptive -- used in get_metrics().
    feature_fn : Callable[[pd.DataFrame], pd.DataFrame]
        Takes raw appointment rows and returns a DataFrame with every
        column in ``feature_cols`` present (plus, harmlessly, anything
        else). This is this model's own feature-engineering step --
        it does not have to match any other model's.
    feature_cols : Sequence[str]
        The exact columns (in the engineered DataFrame) this model was
        trained on, in the order the pipeline expects.
    numeric_cols : Sequence[str]
        The subset of ``feature_cols`` that must be numeric. Validated
        for dtype before scoring.
    """

    def __init__(
        self,
        fitted_pipeline,
        name: str,
        source: str,
        feature_fn: Callable[[pd.DataFrame], pd.DataFrame],
        feature_cols: Sequence[str],
        numeric_cols: Sequence[str],
    ):
        self.pipeline = fitted_pipeline
        self.name = name
        self.source = source
        self.feature_fn = feature_fn
        self.feature_cols = list(feature_cols)
        self.numeric_cols = list(numeric_cols)

    def predict_risk(self, raw_appointments: pd.DataFrame) -> pd.DataFrame:
        """
        Score a batch of raw appointment rows.

        Returns a DataFrame with one row per input row:
        appointment_id, no_show_probability, model_used, risk_band.

        Raises
        ------
        ValueError
            If the input is empty, missing required raw columns, missing
            engineered feature columns, or contains a non-numeric value
            in a column this model treats as numeric.
        RuntimeError
            If the model produces probabilities outside [0, 1], or if the
            output row count doesn't match the input row count -- both
            would indicate a bug in the wrapped pipeline itself, not bad
            input.
        """
        if raw_appointments.empty:
            raise ValueError(f"[{self.name}] No appointment records provided to score.")

        missing_raw = REQUIRED_RAW_COLUMNS - set(raw_appointments.columns)
        if missing_raw:
            raise ValueError(f"[{self.name}] Input is missing required columns: {missing_raw}")

        engineered_batch = self.feature_fn(raw_appointments)

        missing_features = set(self.feature_cols) - set(engineered_batch.columns)
        if missing_features:
            raise ValueError(
                f"[{self.name}] Feature engineering did not produce: {missing_features}"
            )

        for col in self.numeric_cols:
            coerced = pd.to_numeric(engineered_batch[col], errors="coerce")
            bad_rows = coerced.isna() & engineered_batch[col].notna()
            if bad_rows.any():
                bad_values = engineered_batch.loc[bad_rows, col].tolist()
                raise ValueError(
                    f"[{self.name}] Column '{col}' contains non-numeric value(s): {bad_values}"
                )

        logger.info("[%s] Scoring %d appointment(s).", self.name, len(raw_appointments))

        probabilities = self.pipeline.predict_proba(engineered_batch[self.feature_cols])[:, 1]

        if not ((probabilities >= 0) & (probabilities <= 1)).all():
            raise RuntimeError(f"[{self.name}] Model produced probabilities outside [0, 1] range.")

        result = pd.DataFrame(
            {
                "appointment_id": raw_appointments["appointment_id"].values,
                "no_show_probability": probabilities.round(4),
                "model_used": self.name,
            }
        )
        result["risk_band"] = result["no_show_probability"].apply(_risk_band)

        if len(result) != len(raw_appointments):
            raise RuntimeError(f"[{self.name}] Output row count doesn't match input row count.")

        logger.info("[%s] Returned %d risk score(s).", self.name, len(result))
        return result

    def get_metrics(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Return accuracy and ROC-AUC for this model on a held-out set."""
        from sklearn.metrics import accuracy_score, roc_auc_score

        y_pred = self.pipeline.predict(X_test)
        y_proba = self.pipeline.predict_proba(X_test)[:, 1]
        return {
            "model": self.name,
            "source": self.source,
            "accuracy": round(accuracy_score(y_test, y_pred), 3),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 3),
        }


def _risk_band(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Medium"
    return "High"
