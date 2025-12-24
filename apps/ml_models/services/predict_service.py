from __future__ import annotations

import io
import pandas as pd
import joblib

from django.core.files.storage import default_storage
from rest_framework.exceptions import ValidationError

from apps.ml_models.models import Experiment, ModelArtifact


def _load_model_from_artifact(artifact: ModelArtifact):
    if not artifact or not artifact.file:
        raise ValidationError({"model": "Model artifact (.pkl) not found for this experiment."})
    # Ensure local path is available
    model_path = artifact.file.path
    return joblib.load(model_path)


def predict_csv_for_experiment(experiment: Experiment, uploaded_file) -> bytes:
    """
    Load experiment model (.pkl), read uploaded CSV (unlabeled),
    run predict(), and return CSV bytes with a new column: prediction.
    """
    artifact = ModelArtifact.objects.filter(experiment=experiment).first()
    model = _load_model_from_artifact(artifact)

    # Read the uploaded CSV into pandas
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        raise ValidationError({"file": f"Could not read CSV: {str(e)}"})

    if df.shape[0] == 0:
        raise ValidationError({"file": "Uploaded CSV has zero rows."})

    # Predict
    try:
        preds = model.predict(df)
    except Exception as e:
        raise ValidationError({"predict": f"Prediction failed. Ensure columns match training features. Details: {str(e)}"})

    out_df = df.copy()
    out_df["prediction"] = preds

    # Write bytes
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
