from __future__ import annotations

import hashlib
import io

import joblib
from django.core.files.base import ContentFile

from apps.ml_models.models import Experiment, ModelArtifact


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def save_model_artifact(experiment: Experiment, fitted_pipeline) -> ModelArtifact:
    """
    Saves fitted sklearn Pipeline as .pkl using joblib into ModelArtifact.
    """
    buffer = io.BytesIO()
    joblib.dump(fitted_pipeline, buffer)
    data = buffer.getvalue()
    sha256 = _sha256_bytes(data)

    filename = f"{experiment.algorithm.key}_experiment_{experiment.id}.pkl"
    content = ContentFile(data, name=filename)

    obj, _ = ModelArtifact.objects.update_or_create(
        experiment=experiment,
        defaults={
            "sha256": sha256,
            "feature_names": [],
            "target_name": experiment.pipeline.target_column or "",
            "class_labels": [],
        },
    )
    # Save file via FileField
    obj.file.save(filename, content, save=True)
    return obj
