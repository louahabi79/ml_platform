import hashlib
from django.conf import settings
from django.core.validators import FileExtensionValidator
from django.db import models


class Project(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="projects",
        db_index=True,
    )
    name = models.CharField(max_length=140)
    description = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["owner", "name"],
                name="uniq_project_name_per_owner",
            )
        ]
        ordering = ["-updated_at", "-created_at"]

    def __str__(self) -> str:
        return f"{self.name}"


class DatasetSourceType(models.TextChoices):
    CSV_UPLOAD = "CSV_UPLOAD", "CSV Upload"
    MANUAL_TABLE = "MANUAL_TABLE", "Manual Table"
    KAGGLE = "KAGGLE", "Kaggle Import"
    UCI = "UCI", "UCI Import"
    API = "API", "API Import"


class Dataset(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="datasets",
        db_index=True,
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="datasets",
        db_index=True,
    )

    name = models.CharField(max_length=140)
    description = models.TextField(blank=True)

    source_type = models.CharField(
        max_length=20,
        choices=DatasetSourceType.choices,
        db_index=True,
    )

    # Optional metadata for Kaggle/UCI/API “simulation” or real connectors later
    external_source = models.JSONField(
        blank=True,
        default=dict,
        help_text="Metadata for external dataset sources (e.g., dataset id, url, api params).",
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["project", "name"],
                name="uniq_dataset_name_per_project",
            )
        ]
        indexes = [
            models.Index(fields=["owner", "project", "created_at"]),
        ]
        ordering = ["-updated_at", "-created_at"]

    def __str__(self) -> str:
        return f"{self.name}"


class DatasetDraft(models.Model):
    """
    Stores manual table edits from the UI before finalizing into an immutable DatasetVersion.
    Keep drafts reasonably small via app-level limits (rows/cols).
    """
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="dataset_drafts",
        db_index=True,
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="dataset_drafts",
        db_index=True,
    )
    name = models.CharField(max_length=140)

    columns = models.JSONField(default=list, help_text="List of column definitions: [{name, dtype_hint}]")
    rows = models.JSONField(default=list, help_text="List of row dicts. Example: [{'age': 20, 'city': 'LA'}]")

    is_finalized = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [models.Index(fields=["owner", "project", "created_at"])]
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"Draft: {self.name}"


def dataset_version_upload_path(instance: "DatasetVersion", filename: str) -> str:
    return f"datasets/{instance.dataset.owner_id}/{instance.dataset_id}/v{instance.version_number}/{filename}"


class DatasetVersion(models.Model):
    """
    Immutable snapshot of a dataset (for reproducibility).
    Stores file, hash, schema, profiling stats.
    """
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="versions",
        db_index=True,
    )
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="created_dataset_versions",
    )

    version_number = models.PositiveIntegerField()
    file = models.FileField(
        upload_to=dataset_version_upload_path,
        validators=[FileExtensionValidator(["csv"])],
    )

    sha256 = models.CharField(max_length=64, db_index=True)
    row_count = models.PositiveIntegerField(default=0)
    column_count = models.PositiveIntegerField(default=0)

    schema_json = models.JSONField(default=dict, help_text="Column types and roles; stable schema snapshot.")
    profile_json = models.JSONField(default=dict, help_text="Profiling stats: missingness, uniques, etc.")

    notes = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["dataset", "version_number"],
                name="uniq_version_per_dataset",
            )
        ]
        indexes = [
            models.Index(fields=["dataset", "created_at"]),
            models.Index(fields=["sha256"]),
        ]
        ordering = ["-version_number"]

    def __str__(self) -> str:
        return f"{self.dataset.name} v{self.version_number}"

    @staticmethod
    def compute_sha256(content_bytes: bytes) -> str:
        return hashlib.sha256(content_bytes).hexdigest()


class DatasetColumn(models.Model):
    """
    Column-level info for a specific DatasetVersion (supports profiling + UI selection).
    """
    version = models.ForeignKey(
        DatasetVersion,
        on_delete=models.CASCADE,
        related_name="columns",
        db_index=True,
    )

    name = models.CharField(max_length=140)
    pandas_dtype = models.CharField(max_length=64, blank=True)
    inferred_type = models.CharField(
        max_length=32,
        help_text="One of: numeric, categorical, datetime, text, boolean, unknown",
    )

    # Basic stats for UI and quick checks
    missing_count = models.PositiveIntegerField(default=0)
    unique_count = models.PositiveIntegerField(default=0)

    # UI ordering
    position = models.PositiveIntegerField(default=0)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["version", "name"],
                name="uniq_column_name_per_version",
            )
        ]
        indexes = [
            models.Index(fields=["version", "position"]),
        ]
        ordering = ["position", "id"]

    def __str__(self) -> str:
        return f"{self.name} ({self.version})"
