from django.conf import settings
from django.db import models

from apps.datasets.models import Project, DatasetVersion
from apps.ml_models.models import Experiment


class PlotType(models.TextChoices):
    DATASET_DISTRIBUTION = "DATASET_DISTRIBUTION", "Dataset Distribution"
    CORRELATION_MATRIX = "CORRELATION_MATRIX", "Correlation Matrix"
    DECISION_BOUNDARY = "DECISION_BOUNDARY", "Decision Boundary"
    LEARNING_CURVE = "LEARNING_CURVE", "Learning Curve"
    MODEL_COMPARISON = "MODEL_COMPARISON", "Model Comparison"


def plot_upload_path(instance: "PlotArtifact", filename: str) -> str:
    return f"plots/{instance.owner_id}/{instance.project_id}/{instance.plot_type}/{filename}"


class PlotArtifact(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="plots",
        db_index=True,
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="plots",
        db_index=True,
    )

    dataset_version = models.ForeignKey(
        DatasetVersion,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="plots",
    )
    experiment = models.ForeignKey(
        Experiment,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="plots",
    )

    plot_type = models.CharField(max_length=24, choices=PlotType.choices, db_index=True)
    title = models.CharField(max_length=180)

    # PNG output for easy export
    image_file = models.FileField(upload_to=plot_upload_path, blank=True)

    # Optional: store Plotly JSON for interactive rendering
    plotly_json = models.JSONField(default=dict, blank=True)

    config = models.JSONField(default=dict, blank=True, help_text="How the plot was generated (columns, params).")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["owner", "project", "plot_type", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.plot_type}: {self.title}"
