from __future__ import annotations

import importlib
import traceback
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from django.db import transaction
from rest_framework.exceptions import ValidationError

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_validate
from sklearn.base import is_classifier

from apps.ml_models.models import (
    Experiment,
    MLTaskType,
    EvaluationResult,
    CVFoldResult,
)
from apps.ml_models.services.pipeline_builder import build_sklearn_pipeline
from apps.ml_models.services.metrics import classification_metrics, regression_metrics, summarize_cv
from apps.ml_models.services.model_persistence import save_model_artifact

from sklearn.metrics import confusion_matrix
from apps.visualization.services import (
    plot_actual_vs_predicted,
    plot_confusion_matrix_heatmap,
    plot_learning_curve,
)


def _import_class(class_path: str):
    if not class_path or "." not in class_path:
        raise ValidationError({"algorithm.sklearn_class_path": "Invalid sklearn class path."})
    mod, cls = class_path.rsplit(".", 1)
    module = importlib.import_module(mod)
    return getattr(module, cls)


def _instantiate_estimator(experiment: Experiment):
    Estimator = _import_class(experiment.algorithm.sklearn_class_path)
    params = experiment.hyperparameters or {}
    try:
        return Estimator(**params)
    except TypeError as e:
        # If schema allowed something but sklearn rejects (e.g., incompatible solver/penalty)
        raise ValidationError({"hyperparameters": f"Estimator rejected hyperparameters: {str(e)}"})


def _cv_strategy(experiment: Experiment, y):
    folds = int(experiment.cv_folds or 5)
    if folds < 2:
        raise ValidationError({"cv_folds": "cv_folds must be >= 2."})

    if experiment.pipeline.task_type == MLTaskType.CLASSIFICATION:
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=experiment.random_seed)
    if experiment.pipeline.task_type == MLTaskType.REGRESSION:
        return KFold(n_splits=folds, shuffle=True, random_state=experiment.random_seed)

    return None


def _scoring(task_type: str) -> Dict[str, str]:
    if task_type == MLTaskType.CLASSIFICATION:
        return {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    if task_type == MLTaskType.REGRESSION:
        return {"rmse": "neg_root_mean_squared_error", "r2": "r2"}
    return {}


def _postprocess_cv_scores(task_type: str, cv_result: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[int, Dict[str, float]]]:
    """
    Returns:
      - cv_summary: {metric: {mean, std}}
      - per_fold: {fold_idx: {metric: value}}
    """
    cv_summary = {}
    per_fold = {}

    # cross_validate returns keys like test_accuracy, test_f1_macro, test_rmse, etc.
    for key, arr in cv_result.items():
        if not key.startswith("test_"):
            continue

        metric_name = key.replace("test_", "")
        scores = np.array(arr, dtype=float)

        # Convert neg rmse to positive rmse
        if metric_name == "rmse":
            scores = -scores

        cv_summary[metric_name] = summarize_cv(scores)

        for i, v in enumerate(scores):
            per_fold.setdefault(i, {})
            per_fold[i][metric_name] = float(v)

    return cv_summary, per_fold


@transaction.atomic
def _store_cv_folds(experiment: Experiment, per_fold: Dict[int, Dict[str, float]]) -> None:
    CVFoldResult.objects.filter(experiment=experiment).delete()
    for fold_idx, metrics in per_fold.items():
        CVFoldResult.objects.create(
            experiment=experiment,
            fold_index=int(fold_idx),
            metrics=metrics,
        )


@transaction.atomic
def _store_evaluation(experiment: Experiment, metrics: Dict[str, Any], confusion: Dict[str, Any] | None, curves: Dict[str, Any] | None = None) -> EvaluationResult:
    obj, _ = EvaluationResult.objects.update_or_create(
        experiment=experiment,
        defaults={
            "metrics": metrics,
            "confusion_matrix": confusion or {},
            "curves": curves or {},
        },
    )
    return obj


def train_evaluate_and_persist(experiment: Experiment) -> Dict[str, Any]:
    """
    Full experiment run:
      1) Load dataset
      2) Build pipeline from PipelineSteps
      3) Optional CV -> store mean/std + per-fold metrics
      4) Train/test split -> compute test metrics
      5) Refit on full data -> persist .pkl ModelArtifact
      6) Store EvaluationResult
    """
    logs = []
    try:
        dv = experiment.dataset_version
        df = pd.read_csv(dv.file.path)
        logs.append(f"[INFO] Loaded dataset rows={df.shape[0]} cols={df.shape[1]}")

        estimator = _instantiate_estimator(experiment)
        built = build_sklearn_pipeline(experiment.pipeline, df, estimator)
        X, y = built.X, built.y

        if experiment.pipeline.task_type in (MLTaskType.CLASSIFICATION, MLTaskType.REGRESSION) and y is None:
            raise ValidationError({"pipeline": "Supervised task requires target column."})

        # Optional Cross-Validation (on full data)
        cv_payload = {}
        if experiment.use_cross_validation and experiment.pipeline.task_type in (MLTaskType.CLASSIFICATION, MLTaskType.REGRESSION):
            cv = _cv_strategy(experiment, y)
            scoring = _scoring(experiment.pipeline.task_type)
            cv_result = cross_validate(
                built.sklearn_pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=None,
                return_train_score=False,
            )
            cv_summary, per_fold = _postprocess_cv_scores(experiment.pipeline.task_type, cv_result)
            _store_cv_folds(experiment, per_fold)
            cv_payload = {"summary": cv_summary}
            logs.append(f"[OK] Cross-validation complete folds={experiment.cv_folds}")

        # Train/test evaluation
        test_metrics_payload = {}
        confusion_payload = {}

        if experiment.pipeline.task_type == MLTaskType.CLASSIFICATION:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=float(experiment.test_size),
                random_state=experiment.random_seed,
                stratify=y,
            )
            built.sklearn_pipeline.fit(X_train, y_train)
            y_pred = built.sklearn_pipeline.predict(X_test)

            pack = classification_metrics(y_test, y_pred)
            test_metrics_payload = pack.metrics
            confusion_payload = pack.confusion or {}
            logs.append("[OK] Classification train/test evaluation completed.")

            # ---- Plot: confusion matrix heatmap ----
            cm = confusion_matrix(y_test, y_pred)
            # best-effort labels
            class_labels = None
            try:
                model = built.sklearn_pipeline.named_steps["model"]
                if hasattr(model, "classes_"):
                    class_labels = [str(c) for c in model.classes_]
            except Exception:
                class_labels = None

            plot_confusion_matrix_heatmap(
                experiment=experiment,
                cm=cm,
                class_labels=class_labels,
                owner=experiment.owner,
                project=experiment.project,
            )

            # ---- Plot: learning curve (classification scoring) ----
            if experiment.use_cross_validation:
                cv = _cv_strategy(experiment, y)
                plot_learning_curve(
                    experiment=experiment,
                    sklearn_pipeline=built.sklearn_pipeline,
                    X=X,
                    y=y,
                    cv=cv,
                    scoring="accuracy",
                    owner=experiment.owner,
                    project=experiment.project,
                )


        elif experiment.pipeline.task_type == MLTaskType.REGRESSION:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=float(experiment.test_size),
                random_state=experiment.random_seed,
            )
            built.sklearn_pipeline.fit(X_train, y_train)
            y_pred = built.sklearn_pipeline.predict(X_test)

            pack = regression_metrics(y_test, y_pred)
            test_metrics_payload = pack.metrics
            logs.append("[OK] Regression train/test evaluation completed.")

            # ---- Plot: actual vs predicted ----
            plot_actual_vs_predicted(
                experiment=experiment,
                y_true=y_test,
                y_pred=y_pred,
                owner=experiment.owner,
                project=experiment.project,
            )

            # ---- Plot: learning curve (regression scoring) ----
            if experiment.use_cross_validation:
                cv = _cv_strategy(experiment, y)
                plot_learning_curve(
                    experiment=experiment,
                    sklearn_pipeline=built.sklearn_pipeline,
                    X=X,
                    y=y,
                    cv=cv,
                    scoring="r2",
                    owner=experiment.owner,
                    project=experiment.project,
                )


        else:
            # Clustering not required in this phase’s metric requirements; implemented later with inertia/silhouette.
            built.sklearn_pipeline.fit(X)
            logs.append("[OK] Clustering model fit completed (metrics added in later phase).")

        # Refit on full data for final persisted model
        if experiment.pipeline.task_type in (MLTaskType.CLASSIFICATION, MLTaskType.REGRESSION):
            built.sklearn_pipeline.fit(X, y)
            logs.append("[INFO] Refit on full dataset for final model persistence.")
        else:
            built.sklearn_pipeline.fit(X)

        artifact = save_model_artifact(experiment, built.sklearn_pipeline)

        # Attempt to populate feature names / labels after fitting
        try:
            preprocess = built.sklearn_pipeline.named_steps.get("preprocess")
            if preprocess is not None and hasattr(preprocess, "get_feature_names_out"):
                feat = preprocess.get_feature_names_out()
                artifact.feature_names = [str(x) for x in feat]
        except Exception:
            # Non-fatal: feature names are best-effort
            pass

        # Class labels for classification
        if experiment.pipeline.task_type == MLTaskType.CLASSIFICATION and hasattr(built.sklearn_pipeline.named_steps["model"], "classes_"):
            artifact.class_labels = [str(c) for c in built.sklearn_pipeline.named_steps["model"].classes_]

        artifact.save(update_fields=["feature_names", "class_labels"])

        # Store EvaluationResult
        metrics_payload = {
            "test": test_metrics_payload,
        }
        if cv_payload:
            metrics_payload["cv"] = cv_payload["summary"]

        _store_evaluation(
            experiment=experiment,
            metrics=metrics_payload,
            confusion=confusion_payload if confusion_payload else {},
            curves={},
        )

        return {
            "ok": True,
            "model_artifact_id": artifact.id,
            "metrics": metrics_payload,
            "logs": "\n".join(logs) + "\n",
        }

    except Exception as e:
        tb = traceback.format_exc()
        return {
            "ok": False,
            "error": str(e),
            "traceback": tb,
            "logs": "\n".join(logs) + "\n",
        }
