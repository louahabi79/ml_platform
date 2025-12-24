from __future__ import annotations

from django.conf import settings
from django.core.validators import MinValueValidator
from django.db import models

from apps.datasets.models import Project, DatasetVersion


class MLTaskType(models.TextChoices):
    REGRESSION = "REGRESSION", "Regression"
    CLASSIFICATION = "CLASSIFICATION", "Classification"
    CLUSTERING = "CLUSTERING", "Clustering"


class AlgorithmFamily(models.TextChoices):
    REGRESSION = "REGRESSION", "Regression"
    CLASSIFICATION = "CLASSIFICATION", "Classification"
    CLUSTERING = "CLUSTERING", "Clustering"
    NEURAL_NETWORK = "NEURAL_NETWORK", "Neural Network"


class Algorithm(models.Model):
    """
    Registry for supported algorithms (required for in-app tutorials + UI parameter forms).
    """
    key = models.SlugField(
        max_length=60,
        unique=True,
        help_text="Stable identifier (e.g., 'logistic-regression', 'random-forest').",
    )
    display_name = models.CharField(max_length=120)
    family = models.CharField(max_length=20, choices=AlgorithmFamily.choices, db_index=True)

    # How we instantiate it in code:
    # Example: 'sklearn.linear_model.LogisticRegression'
    sklearn_class_path = models.CharField(max_length=200, blank=True)

    # Which tasks it can run
    supported_tasks = models.JSONField(
        default=list,
        help_text="List like ['CLASSIFICATION'] or ['REGRESSION','CLASSIFICATION'] depending on algorithm usage.",
    )

    # Hyperparameter support for UI:
    default_hyperparameters = models.JSONField(default=dict)
    hyperparameter_schema = models.JSONField(
        default=dict,
        help_text="JSON schema to drive dynamic forms: field types, ranges, enums, help text.",
    )

    # ✅ Educational content (in-app tutorials)
    educational_summary = models.TextField(
        help_text="Short student-friendly summary of how it works.",
    )
    educational_details_md = models.TextField(
        help_text="Longer explanation in Markdown (math intuition + steps).",
    )
    prerequisites_md = models.TextField(
        blank=True,
        help_text="Background knowledge (e.g., vectors, probability, gradients). Markdown.",
    )
    typical_use_cases_md = models.TextField(
        blank=True,
        help_text="When to use / when not to use. Markdown.",
    )
    strengths_md = models.TextField(blank=True)
    limitations_md = models.TextField(blank=True)
    references = models.JSONField(
        default=list,
        help_text="List of references: [{title, author, year, link}]",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["family", "display_name"]

    def __str__(self) -> str:
        return self.display_name


class AlgorithmTutorialSection(models.Model):
    """
    Structured tutorial sections for each algorithm (supports multi-page learning inside the app).
    """
    algorithm = models.ForeignKey(
        Algorithm,
        on_delete=models.CASCADE,
        related_name="tutorial_sections",
        db_index=True,
    )
    title = models.CharField(max_length=140)
    order = models.PositiveIntegerField(default=0)

    content_md = models.TextField(help_text="Markdown tutorial content.")
    quiz_questions = models.JSONField(
        default=list,
        help_text="Optional: list of quiz questions for students (MCQ/short).",
        blank=True,
    )

    class Meta:
        ordering = ["order", "id"]
        indexes = [models.Index(fields=["algorithm", "order"])]

    def __str__(self) -> str:
        return f"{self.algorithm.display_name}: {self.title}"


class Pipeline(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="pipelines",
        db_index=True,
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="pipelines",
        db_index=True,
    )
    dataset_version = models.ForeignKey(
        DatasetVersion,
        on_delete=models.CASCADE,
        related_name="pipelines",
        db_index=True,
    )

    name = models.CharField(max_length=140)
    task_type = models.CharField(max_length=20, choices=MLTaskType.choices, db_index=True)

    target_column = models.CharField(max_length=140, blank=True, help_text="Required for supervised tasks.")
    feature_columns = models.JSONField(
        default=list,
        help_text="List of selected feature column names; empty means 'all except target'.",
    )

    random_seed = models.PositiveIntegerField(default=42)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["project", "name"],
                name="uniq_pipeline_name_per_project",
            )
        ]
        indexes = [
            models.Index(fields=["owner", "project", "dataset_version"]),
        ]
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return self.name


class PipelineStepType(models.TextChoices):
    MISSING_VALUES = "MISSING_VALUES", "Missing Values"
    ENCODING = "ENCODING", "Encoding"
    SCALING = "SCALING", "Scaling"
    FEATURE_SELECTION = "FEATURE_SELECTION", "Feature Selection"
    POLYNOMIAL_FEATURES = "POLYNOMIAL_FEATURES", "Polynomial Features"
    MANUAL_FEATURES = "MANUAL_FEATURES", "Manual Features"


class PipelineStep(models.Model):
    pipeline = models.ForeignKey(
        Pipeline,
        on_delete=models.CASCADE,
        related_name="steps",
        db_index=True,
    )
    step_type = models.CharField(max_length=24, choices=PipelineStepType.choices, db_index=True)
    order = models.PositiveIntegerField(default=0)
    enabled = models.BooleanField(default=True)

    config = models.JSONField(
        default=dict,
        help_text="Step configuration (e.g., strategy='mean', scaler='standard', encoder='onehot').",
    )

    class Meta:
        ordering = ["order", "id"]
        indexes = [models.Index(fields=["pipeline", "order"])]
        constraints = [
            models.UniqueConstraint(
                fields=["pipeline", "order"],
                name="uniq_step_order_per_pipeline",
            )
        ]

    def __str__(self) -> str:
        return f"{self.pipeline.name} - {self.step_type} ({self.order})"


class ManualFeature(models.Model):
    """
    User-defined feature formulas.
    Example expression: 'income / (age + 1)'.
    We will validate and safely evaluate expressions in service layer later.
    """
    pipeline = models.ForeignKey(
        Pipeline,
        on_delete=models.CASCADE,
        related_name="manual_features",
        db_index=True,
    )

    output_name = models.CharField(max_length=140)
    expression = models.CharField(max_length=500)
    description = models.TextField(blank=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["pipeline", "output_name"],
                name="uniq_manual_feature_name_per_pipeline",
            )
        ]

    def __str__(self) -> str:
        return self.output_name


class ExperimentStatus(models.TextChoices):
    PENDING = "PENDING", "Pending"
    RUNNING = "RUNNING", "Running"
    SUCCEEDED = "SUCCEEDED", "Succeeded"
    FAILED = "FAILED", "Failed"


class Experiment(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="experiments",
        db_index=True,
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="experiments",
        db_index=True,
    )
    dataset_version = models.ForeignKey(
        DatasetVersion,
        on_delete=models.CASCADE,
        related_name="experiments",
        db_index=True,
    )
    pipeline = models.ForeignKey(
        Pipeline,
        on_delete=models.CASCADE,
        related_name="experiments",
        db_index=True,
    )
    algorithm = models.ForeignKey(
        Algorithm,
        on_delete=models.PROTECT,
        related_name="experiments",
        db_index=True,
    )

    name = models.CharField(max_length=160)
    notes = models.TextField(blank=True)

    # Hyperparameters for this run
    hyperparameters = models.JSONField(default=dict)

    # Validation config
    test_size = models.FloatField(default=0.2, validators=[MinValueValidator(0.0)])
    use_cross_validation = models.BooleanField(default=True)
    cv_folds = models.PositiveIntegerField(default=5)

    random_seed = models.PositiveIntegerField(default=42)

    status = models.CharField(max_length=12, choices=ExperimentStatus.choices, default=ExperimentStatus.PENDING)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    error_message = models.TextField(blank=True)
    run_logs = models.TextField(blank=True)

    # Reproducibility metadata (stored at run time)
    runtime_meta = models.JSONField(
        default=dict,
        help_text="Versions: python, sklearn, pandas; dataset hash; etc.",
    )

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["owner", "project", "status", "created_at"]),
            models.Index(fields=["algorithm", "created_at"]),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=["project", "name"],
                name="uniq_experiment_name_per_project",
            )
        ]
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name


def model_artifact_upload_path(instance: "ModelArtifact", filename: str) -> str:
    return f"models/{instance.experiment.owner_id}/{instance.experiment_id}/{filename}"


class ModelArtifact(models.Model):
    """
    Persisted sklearn Pipeline/Estimator + preprocessing via joblib.
    """
    experiment = models.OneToOneField(
        Experiment,
        on_delete=models.CASCADE,
        related_name="model_artifact",
    )

    file = models.FileField(upload_to=model_artifact_upload_path)
    sha256 = models.CharField(max_length=64, db_index=True)

    feature_names = models.JSONField(default=list)
    target_name = models.CharField(max_length=140, blank=True)
    class_labels = models.JSONField(default=list, blank=True)

    trained_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=["sha256"])]

    def __str__(self) -> str:
        return f"Model for {self.experiment.name}"


class EvaluationResult(models.Model):
    """
    Stores computed metrics and artifacts for one experiment.
    Supports both supervised + clustering.
    """
    experiment = models.OneToOneField(
        Experiment,
        on_delete=models.CASCADE,
        related_name="evaluation",
    )

    # Generic metrics store:
    # - classification: accuracy, precision, recall, f1, etc.
    # - regression: rmse, mae, r2
    # - clustering: inertia, silhouette (if computed)
    metrics = models.JSONField(default=dict)

    # Confusion matrix (classification)
    confusion_matrix = models.JSONField(default=dict, blank=True)

    # Learning curve points, ROC points, etc.
    curves = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"Evaluation: {self.experiment.name}"


class CVFoldResult(models.Model):
    experiment = models.ForeignKey(
        Experiment,
        on_delete=models.CASCADE,
        related_name="cv_fold_results",
        db_index=True,
    )
    fold_index = models.PositiveIntegerField()
    metrics = models.JSONField(default=dict)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["experiment", "fold_index"],
                name="uniq_fold_per_experiment",
            )
        ]
        ordering = ["fold_index"]

    def __str__(self) -> str:
        return f"{self.experiment.name} fold {self.fold_index}"


def prediction_upload_path(instance: "PredictionBatch", filename: str) -> str:
    return f"predictions/{instance.owner_id}/{instance.experiment_id}/{filename}"


class PredictionBatch(models.Model):
    """
    Stores prediction outputs for an experiment.
    """
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="prediction_batches",
        db_index=True,
    )
    experiment = models.ForeignKey(
        Experiment,
        on_delete=models.CASCADE,
        related_name="predictions",
        db_index=True,
    )

    input_meta = models.JSONField(
        default=dict,
        help_text="Info about input rows/source used for prediction.",
    )
    output_file = models.FileField(upload_to=prediction_upload_path)

    summary = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Predictions for {self.experiment.name}"


class ExportType(models.TextChoices):
    MODEL_PKL = "MODEL_PKL", "Model (.pkl)"
    REPORT_PDF = "REPORT_PDF", "Evaluation Report (PDF)"
    REPORT_CSV = "REPORT_CSV", "Evaluation Report (CSV)"
    PLOT_PNG = "PLOT_PNG", "Visualization (PNG)"
    PREDICTIONS_CSV = "PREDICTIONS_CSV", "Predictions (CSV)"
    BUNDLE_ZIP = "BUNDLE_ZIP", "Bundle (ZIP)"


def export_upload_path(instance: "ExportArtifact", filename: str) -> str:
    return f"exports/{instance.owner_id}/{instance.project_id}/{instance.export_type}/{filename}"


class ExportArtifact(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="exports",
        db_index=True,
    )
    project = models.ForeignKey(
        Project,
        on_delete=models.CASCADE,
        related_name="exports",
        db_index=True,
    )

    experiment = models.ForeignKey(
        Experiment,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="exports",
    )

    export_type = models.CharField(max_length=20, choices=ExportType.choices, db_index=True)
    file = models.FileField(upload_to=export_upload_path)
    meta = models.JSONField(default=dict, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["owner", "project", "export_type", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.export_type} ({self.project.name})"
