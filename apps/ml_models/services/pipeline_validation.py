from __future__ import annotations

from rest_framework.exceptions import ValidationError

from apps.ml_models.models import Pipeline, PipelineStep, PipelineStepType, MLTaskType


def validate_pipeline_or_raise(pipeline: Pipeline) -> None:
    """
    Validates pipeline consistency before training.
    Keeps the View thin.
    """
    if pipeline.task_type in (MLTaskType.REGRESSION, MLTaskType.CLASSIFICATION):
        if not pipeline.target_column:
            raise ValidationError({"pipeline.target_column": "Target column is required for supervised tasks."})

    # Ensure steps have unique order and are contiguous-ish (optional)
    steps = list(PipelineStep.objects.filter(pipeline=pipeline).order_by("order"))
    orders = [s.order for s in steps]
    if len(orders) != len(set(orders)):
        raise ValidationError({"pipeline.steps": "Duplicate step order detected."})

    # Example: encoding makes sense if categorical features exist; we don’t block here,
    # but we ensure the config is not empty if enabled.
    for s in steps:
        if s.enabled and s.step_type != PipelineStepType.MANUAL_FEATURES and s.config is None:
            raise ValidationError({f"pipeline.step[{s.order}].config": "Config must be provided for enabled step."})
