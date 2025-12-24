from __future__ import annotations

import tempfile

import pandas as pd
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from sklearn.datasets import make_classification

from apps.datasets.models import Project, Dataset, DatasetVersion
from apps.ml_models.models import (
    Algorithm,
    Pipeline,
    PipelineStep,
    PipelineStepType,
    Experiment,
    MLTaskType,
    EvaluationResult,
    ModelArtifact,
)
from apps.ml_models.services.experiment_runner import train_evaluate_and_persist
from apps.visualization.models import PlotArtifact


User = get_user_model()


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(prefix="ml_platform_test_media_"))
class ExperimentRunnerE2ETests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="runner", password="pass12345")
        self.project = Project.objects.create(owner=self.user, name="Proj", description="E2E test")

        self.dataset = Dataset.objects.create(
            owner=self.user,
            project=self.project,
            name="Dummy Classification",
            description="Generated dataset for runner tests",
            source_type="CSV_UPLOAD",
            external_source={},
        )

        X, y = make_classification(
            n_samples=60,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            random_state=42,
        )

        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        df["target"] = y

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        upload = SimpleUploadedFile("dummy.csv", csv_bytes, content_type="text/csv")

        sha256 = DatasetVersion.compute_sha256(csv_bytes)
        self.dv = DatasetVersion.objects.create(
            dataset=self.dataset,
            created_by=self.user,
            version_number=1,
            file=upload,
            sha256=sha256,
            row_count=df.shape[0],
            column_count=df.shape[1],
            schema_json={"columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]},
            profile_json={},
            notes="",
        )

        # Algorithm: Logistic Regression
        self.algorithm = Algorithm.objects.create(
            key="logreg_test",
            display_name="Logistic Regression",
            family="classification",
            sklearn_class_path="sklearn.linear_model.LogisticRegression",
            supported_tasks=["classification"],
            default_hyperparameters={"max_iter": 200},
            hyperparameter_schema={
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "max_iter": {"type": "integer", "minimum": 50, "maximum": 500, "default": 200},
                    "C": {"type": "number", "minimum": 0.0001, "default": 1.0},
                    "solver": {"type": "string", "enum": ["lbfgs", "liblinear", "saga"], "default": "lbfgs"},
                },
            },
            educational_summary="Test algorithm",
            educational_details_md="",
            prerequisites_md="",
            typical_use_cases_md="",
            strengths_md="",
            limitations_md="",
            references=[],
        )

        self.pipeline = Pipeline.objects.create(
            owner=self.user,
            project=self.project,
            dataset_version=self.dv,
            name="Baseline Pipeline",
            task_type=MLTaskType.CLASSIFICATION,
            target_column="target",
            feature_columns=[f"f{i}" for i in range(6)],
            random_seed=42,
        )

        # Optional: scaling (keeps pipeline realistic)
        PipelineStep.objects.create(
            pipeline=self.pipeline,
            step_type=PipelineStepType.SCALING,
            order=0,
            enabled=True,
            config={"scaler": "standard", "columns": []},
        )

        self.experiment = Experiment.objects.create(
            owner=self.user,
            project=self.project,
            dataset_version=self.dv,
            pipeline=self.pipeline,
            algorithm=self.algorithm,
            name="E2E Logistic Regression",
            notes="",
            hyperparameters={"max_iter": 200, "solver": "lbfgs", "C": 1.0},
            test_size=0.25,
            use_cross_validation=False,  # keep test fast; confusion plot still generated
            cv_folds=5,
            random_seed=42,
            status="PENDING",
            run_logs="",
            runtime_meta={},
        )

    def test_runner_creates_metrics_and_artifacts(self):
        result = train_evaluate_and_persist(self.experiment)
        self.assertTrue(result["ok"], msg=result.get("traceback", ""))

        # EvaluationResult created
        ev = EvaluationResult.objects.filter(experiment=self.experiment).first()
        self.assertIsNotNone(ev, "EvaluationResult should be created.")
        self.assertIn("test", ev.metrics)
        self.assertIn("accuracy", ev.metrics["test"])
        self.assertIn("f1_macro", ev.metrics["test"])

        # ModelArtifact created and has a file
        art = ModelArtifact.objects.filter(experiment=self.experiment).first()
        self.assertIsNotNone(art, "ModelArtifact should be created.")
        self.assertTrue(bool(getattr(art, "file", None)), "ModelArtifact must store the .pkl file.")
        self.assertTrue(art.file.name.endswith(".pkl"))

        # PlotArtifact created (confusion matrix heatmap)
        plots = PlotArtifact.objects.filter(experiment=self.experiment)
        self.assertTrue(plots.exists(), "At least one PlotArtifact should be created for classification.")
