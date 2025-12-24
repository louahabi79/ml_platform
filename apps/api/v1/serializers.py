from __future__ import annotations

from rest_framework import serializers

from apps.datasets.models import Project, Dataset, DatasetVersion
from apps.ml_models.models import (
    Algorithm,
    AlgorithmTutorialSection,
    Pipeline,
    PipelineStep,
    PipelineStepType,
    Experiment,
)
from apps.ml_models.services.schema_validation import validate_json_schema_or_raise


# ---------------------------
# Algorithm (read-only)
# ---------------------------

class AlgorithmTutorialSectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlgorithmTutorialSection
        fields = ["id", "title", "order", "content_md", "quiz_questions"]


class AlgorithmSerializer(serializers.ModelSerializer):
    tutorial_sections = AlgorithmTutorialSectionSerializer(many=True, read_only=True)

    class Meta:
        model = Algorithm
        fields = [
            "id",
            "key",
            "display_name",
            "family",
            "sklearn_class_path",
            "supported_tasks",
            "default_hyperparameters",
            "hyperparameter_schema",
            "educational_summary",
            "educational_details_md",
            "prerequisites_md",
            "typical_use_cases_md",
            "strengths_md",
            "limitations_md",
            "references",
            "tutorial_sections",
        ]
        read_only_fields = fields


# ---------------------------
# Projects / Datasets
# ---------------------------

class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = ["id", "name", "description", "created_at", "updated_at"]
        read_only_fields = ["id", "created_at", "updated_at"]


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = [
            "id", "project", "name", "description", "source_type",
            "external_source", "created_at", "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class DatasetVersionSerializer(serializers.ModelSerializer):
    class Meta:
        model = DatasetVersion
        fields = [
            "id",
            "dataset",
            "version_number",
            "file",
            "sha256",
            "row_count",
            "column_count",
            "schema_json",
            "profile_json",
            "notes",
            "created_at",
        ]
        read_only_fields = [
            "id", "version_number", "sha256", "row_count", "column_count",
            "schema_json", "profile_json", "created_at",
        ]


# ---------------------------
# Pipeline + polymorphic steps
# ---------------------------

class PipelineSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pipeline
        fields = [
            "id",
            "project",
            "dataset_version",
            "name",
            "task_type",
            "target_column",
            "feature_columns",
            "random_seed",
            "created_at",
            "updated_at",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


# ---- Step config serializers (polymorphism) ----

class MissingValuesConfigSerializer(serializers.Serializer):
    strategy = serializers.ChoiceField(
        choices=["mean", "median", "most_frequent", "constant", "drop_rows", "drop_columns"]
    )
    fill_value = serializers.JSONField(required=False, allow_null=True)
    columns = serializers.ListField(child=serializers.CharField(), required=False, allow_empty=True)

    def validate(self, attrs):
        if attrs["strategy"] == "constant" and "fill_value" not in attrs:
            raise serializers.ValidationError({"fill_value": "Required when strategy='constant'."})
        return attrs


class EncodingConfigSerializer(serializers.Serializer):
    encoder = serializers.ChoiceField(choices=["onehot", "ordinal"])
    handle_unknown = serializers.ChoiceField(choices=["ignore", "error"], required=False, default="ignore")
    columns = serializers.ListField(child=serializers.CharField(), required=False, allow_empty=True)


class ScalingConfigSerializer(serializers.Serializer):
    scaler = serializers.ChoiceField(choices=["standard", "minmax", "robust", "none"])
    columns = serializers.ListField(child=serializers.CharField(), required=False, allow_empty=True)


class FeatureSelectionConfigSerializer(serializers.Serializer):
    method = serializers.ChoiceField(choices=["variance_threshold", "k_best", "model_based", "none"])
    threshold = serializers.FloatField(required=False)
    k = serializers.IntegerField(required=False, min_value=1)
    score_func = serializers.ChoiceField(
        choices=["f_classif", "mutual_info_classif", "f_regression", "mutual_info_regression"],
        required=False,
    )

    def validate(self, attrs):
        method = attrs["method"]
        if method == "variance_threshold" and "threshold" not in attrs:
            raise serializers.ValidationError({"threshold": "Required for variance_threshold."})
        if method == "k_best" and ("k" not in attrs or "score_func" not in attrs):
            raise serializers.ValidationError({"k": "Required for k_best.", "score_func": "Required for k_best."})
        return attrs


class PolynomialFeaturesConfigSerializer(serializers.Serializer):
    degree = serializers.IntegerField(min_value=2, max_value=10, default=2)
    include_bias = serializers.BooleanField(default=False)
    interaction_only = serializers.BooleanField(default=False)


class ManualFeaturesConfigSerializer(serializers.Serializer):
    # The manual formulas are stored in ManualFeature model; this step config just toggles application.
    enabled = serializers.BooleanField(default=True)


STEP_CONFIG_SERIALIZERS = {
    PipelineStepType.MISSING_VALUES: MissingValuesConfigSerializer,
    PipelineStepType.ENCODING: EncodingConfigSerializer,
    PipelineStepType.SCALING: ScalingConfigSerializer,
    PipelineStepType.FEATURE_SELECTION: FeatureSelectionConfigSerializer,
    PipelineStepType.POLYNOMIAL_FEATURES: PolynomialFeaturesConfigSerializer,
    PipelineStepType.MANUAL_FEATURES: ManualFeaturesConfigSerializer,
}


class PipelineStepSerializer(serializers.ModelSerializer):
    """
    Polymorphic serializer: validates `config` using a step-specific serializer.
    """
    config = serializers.JSONField()

    class Meta:
        model = PipelineStep
        fields = ["id", "pipeline", "step_type", "order", "enabled", "config"]
        read_only_fields = ["id"]

    def validate(self, attrs):
        step_type = attrs.get("step_type", getattr(self.instance, "step_type", None))
        config = attrs.get("config", getattr(self.instance, "config", {}))

        if step_type not in STEP_CONFIG_SERIALIZERS:
            raise serializers.ValidationError({"step_type": f"Unsupported step_type: {step_type}"})

        serializer_cls = STEP_CONFIG_SERIALIZERS[step_type]
        ser = serializer_cls(data=config)
        ser.is_valid(raise_exception=True)
        attrs["config"] = ser.validated_data
        return attrs


# ---------------------------
# Experiment creation: validate hyperparameters via Algorithm schema
# ---------------------------

class ExperimentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Experiment
        fields = [
            "id",
            "project",
            "dataset_version",
            "pipeline",
            "algorithm",
            "name",
            "notes",
            "hyperparameters",
            "test_size",
            "use_cross_validation",
            "cv_folds",
            "random_seed",
            "status",
            "started_at",
            "finished_at",
            "error_message",
            "run_logs",
            "runtime_meta",
            "created_at",
        ]
        read_only_fields = [
            "id", "status", "started_at", "finished_at",
            "error_message", "run_logs", "runtime_meta", "created_at",
        ]

    def validate(self, attrs):
        """
        Critical requirement: validate hyperparameters against algorithm.hyperparameter_schema.
        """
        algorithm: Algorithm = attrs["algorithm"]
        hyperparams = attrs.get("hyperparameters") or {}

        validate_json_schema_or_raise(
            schema=algorithm.hyperparameter_schema,
            instance=hyperparams,
            context_prefix="hyperparameters",
        )
        return attrs
