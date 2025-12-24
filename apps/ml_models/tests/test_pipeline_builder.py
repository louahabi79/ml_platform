from __future__ import annotations

import io
import os
import tempfile

import pandas as pd
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

from apps.datasets.models import Project, Dataset, DatasetVersion
from apps.ml_models.models import Pipeline, PipelineStep, PipelineStepType, MLTaskType
from apps.ml_models.services.pipeline_builder import build_sklearn_pipeline


User = get_user_model()


@override_settings(MEDIA_ROOT=tempfile.mkdtemp(prefix="ml_platform_test_media_"))
class PipelineBuilderTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="tester", password="pass12345")
        self.project = Project.objects.create(owner=self.user, name="P1", description="Test project")

        self.dataset = Dataset.objects.create(
            owner=self.user,
            project=self.project,
            name="Numeric Dataset",
            description="All numeric to isolate scaling behavior",
            source_type="CSV_UPLOAD",
            external_source={},
        )

        # Create a simple numeric dataset
        df = pd.DataFrame({
            "x1": [1.0, 2.0, 3.0, 4.0],
            "x2": [10.0, 20.0, 10.0, 30.0],
            "target": [0, 1, 0, 1],
        })
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        upload = SimpleUploadedFile("data.csv", csv_bytes, content_type="text/csv")

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

        self.pipeline = Pipeline.objects.create(
            owner=self.user,
            project=self.project,
            dataset_version=self.dv,
            name="Scaling Pipeline",
            task_type=MLTaskType.CLASSIFICATION,
            target_column="target",
            feature_columns=["x1", "x2"],
            random_seed=42,
        )

        # Add SCALING step -> StandardScaler
        PipelineStep.objects.create(
            pipeline=self.pipeline,
            step_type=PipelineStepType.SCALING,
            order=0,
            enabled=True,
            config={"scaler": "standard", "columns": []},
        )

    def test_scaling_step_adds_standard_scaler(self):
        df = pd.read_csv(self.dv.file.path)

        estimator = LogisticRegression(max_iter=200)
        built = build_sklearn_pipeline(self.pipeline, df, estimator)
        skl_pipe = built.sklearn_pipeline

        self.assertIn("preprocess", skl_pipe.named_steps)
        preprocess = skl_pipe.named_steps["preprocess"]
        self.assertIsInstance(preprocess, ColumnTransformer)

        # Find numeric transformer
        transformers = dict((name, trans) for name, trans, cols in preprocess.transformers)
        self.assertIn("num", transformers)

        num_transformer = transformers["num"]
        self.assertIsInstance(num_transformer, SkPipeline)

        # Ensure StandardScaler is present
        self.assertIn("scaler", num_transformer.named_steps)
        self.assertIsInstance(num_transformer.named_steps["scaler"], StandardScaler)
