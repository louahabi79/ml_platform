from __future__ import annotations

import io
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from django.core.files.base import ContentFile
from django.utils import timezone

from sklearn.model_selection import learning_curve

from apps.visualization.models import PlotArtifact
from apps.ml_models.models import Experiment, MLTaskType, EvaluationResult
from apps.datasets.models import DatasetVersion


def _save_png_to_plot(plot: PlotArtifact, filename: str, png_bytes: bytes) -> None:
    content = ContentFile(png_bytes, name=filename)
    if hasattr(plot, "image_file"):
        plot.image_file.save(filename, content, save=True)
    elif hasattr(plot, "image"):
        plot.image.save(filename, content, save=True)
    elif hasattr(plot, "file"):
        plot.file.save(filename, content, save=True)
    else:
        raise AttributeError("PlotArtifact must have 'image_file', 'image', or 'file'.")

def _fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _upsert_plot(
    *,
    owner,
    project,
    plot_type: str,
    title: str,
    dataset_version: Optional[DatasetVersion] = None,
    experiment: Optional[Experiment] = None,
    metadata: Optional[Dict[str, Any]] = None,
    png_bytes: bytes,
) -> PlotArtifact:
    """
    Supports 3 scopes:
    - Experiment plot: keyed by (experiment, plot_type)
    - DatasetVersion plot: keyed by (dataset_version, plot_type) where experiment is null
    - Project plot: keyed by (project, plot_type) where both experiment and dataset_version are null
    """
    metadata = metadata or {}

    defaults = {
        "owner": owner,
        "project": project,
        "title": title,
        "config": metadata,
        "created_at": timezone.now(),
    }

    if experiment is not None:
        plot, _ = PlotArtifact.objects.update_or_create(
            experiment=experiment,
            plot_type=plot_type,
            defaults={**defaults, "dataset_version": dataset_version},
        )
        fname = f"exp_{experiment.id}_{plot_type}.png"

    elif dataset_version is not None:
        plot, _ = PlotArtifact.objects.update_or_create(
            dataset_version=dataset_version,
            experiment=None,
            plot_type=plot_type,
            defaults={**defaults},
        )
        fname = f"dv_{dataset_version.id}_{plot_type}.png"

    else:
        # Project-level plot
        plot, _ = PlotArtifact.objects.update_or_create(
            project=project,
            experiment=None,
            dataset_version=None,
            plot_type=plot_type,
            defaults={**defaults},
        )
        fname = f"project_{project.id}_{plot_type}.png"

    _save_png_to_plot(plot, fname, png_bytes)
    return plot


# ------------------------ Dataset Plots ------------------------

def plot_numeric_distributions(
    *,
    dataset_version: DatasetVersion,
    df: pd.DataFrame,
    owner,
    project,
    max_cols: int = 12,
) -> Optional[PlotArtifact]:
    num_cols = list(df.select_dtypes(include=["number", "bool"]).columns)
    if not num_cols:
        return None

    num_cols = num_cols[:max_cols]
    n = len(num_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(num_cols):
        ax = axes[i]
        series = df[col].dropna()
        ax.hist(series.values, bins=25)
        ax.set_title(f"Distribution: {col}", fontsize=10)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    png = _fig_to_png_bytes(fig)
    return _upsert_plot(
        owner=owner,
        project=project,
        dataset_version=dataset_version,
        plot_type="dataset_distributions",
        title="Dataset Distributions (Numeric)",
        metadata={"numeric_columns": num_cols},
        png_bytes=png,
    )


def plot_correlation_heatmap(
    *,
    dataset_version: DatasetVersion,
    df: pd.DataFrame,
    owner,
    project,
    max_cols: int = 20,
) -> Optional[PlotArtifact]:
    num_df = df.select_dtypes(include=["number", "bool"])
    if num_df.shape[1] < 2:
        return None

    if num_df.shape[1] > max_cols:
        num_df = num_df.iloc[:, :max_cols]

    corr = num_df.corr(numeric_only=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    sns.heatmap(corr, ax=ax, annot=False, square=True)
    ax.set_title("Correlation Heatmap (Numeric Features)")

    png = _fig_to_png_bytes(fig)
    return _upsert_plot(
        owner=owner,
        project=project,
        dataset_version=dataset_version,
        plot_type="dataset_correlation",
        title="Correlation Heatmap",
        metadata={"columns": list(num_df.columns)},
        png_bytes=png,
    )


def generate_dataset_plots(dataset_version: DatasetVersion, df: pd.DataFrame, owner, project) -> Dict[str, Any]:
    outputs = {"created": []}
    p1 = plot_numeric_distributions(dataset_version=dataset_version, df=df, owner=owner, project=project)
    if p1:
        outputs["created"].append(p1.id)

    p2 = plot_correlation_heatmap(dataset_version=dataset_version, df=df, owner=owner, project=project)
    if p2:
        outputs["created"].append(p2.id)

    return outputs


# ------------------------ Experiment Plots ------------------------

def plot_actual_vs_predicted(
    *,
    experiment: Experiment,
    y_true,
    y_pred,
    owner,
    project,
) -> PlotArtifact:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, alpha=0.6)
    min_v = float(np.min([y_true.min(), y_pred.min()]))
    max_v = float(np.max([y_true.max(), y_pred.max()]))
    ax.plot([min_v, max_v], [min_v, max_v], linestyle="--")
    ax.set_title("Actual vs Predicted")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    png = _fig_to_png_bytes(fig)
    return _upsert_plot(
        owner=owner,
        project=project,
        experiment=experiment,
        dataset_version=experiment.dataset_version,
        plot_type="exp_actual_vs_predicted",
        title="Actual vs Predicted (Regression)",
        metadata={},
        png_bytes=png,
    )


def plot_confusion_matrix_heatmap(
    *,
    experiment: Experiment,
    cm: np.ndarray,
    class_labels: Optional[List[str]],
    owner,
    project,
) -> PlotArtifact:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax, cbar=False)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    if class_labels and len(class_labels) == cm.shape[0]:
        ax.set_xticklabels(class_labels, rotation=45, ha="right")
        ax.set_yticklabels(class_labels, rotation=0)

    png = _fig_to_png_bytes(fig)
    return _upsert_plot(
        owner=owner,
        project=project,
        experiment=experiment,
        dataset_version=experiment.dataset_version,
        plot_type="exp_confusion_matrix",
        title="Confusion Matrix (Classification)",
        metadata={"class_labels": class_labels or []},
        png_bytes=png,
    )


def plot_learning_curve(
    *,
    experiment: Experiment,
    sklearn_pipeline,
    X,
    y,
    cv,
    scoring: str,
    owner,
    project,
) -> PlotArtifact:
    train_sizes = np.linspace(0.1, 1.0, 5)

    sizes, train_scores, val_scores = learning_curve(
        sklearn_pipeline,
        X,
        y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=None,
        shuffle=True,
        random_state=experiment.random_seed,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1, ddof=1) if train_scores.shape[1] > 1 else np.zeros_like(train_mean)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1, ddof=1) if val_scores.shape[1] > 1 else np.zeros_like(val_mean)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.plot(sizes, train_mean, marker="o", label="Train")
    ax.plot(sizes, val_mean, marker="o", label="Validation")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    ax.set_title(f"Learning Curve ({scoring})")
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.legend()

    png = _fig_to_png_bytes(fig)
    return _upsert_plot(
        owner=owner,
        project=project,
        experiment=experiment,
        dataset_version=experiment.dataset_version,
        plot_type="exp_learning_curve",
        title="Learning Curve",
        metadata={"scoring": scoring},
        png_bytes=png,
    )


# ------------------------ Project Comparison Plot (REQUIRED) ------------------------

def plot_project_model_comparison(*, project, owner) -> Optional[PlotArtifact]:
    """
    Bar chart comparing experiments in a project by primary test metric:
      - Classification -> Accuracy
      - Regression -> R2
    Stores a project-level PlotArtifact: (project, plot_type='project_model_comparison', experiment=None, dataset_version=None)
    """
    exps = (
        Experiment.objects.filter(project=project, owner=owner)
        .select_related("pipeline", "algorithm")
        .order_by("created_at")
    )
    labels: List[str] = []
    values: List[float] = []
    metric_name = None

    for e in exps:
        ev = EvaluationResult.objects.filter(experiment=e).first()
        if not ev or not isinstance(ev.metrics, dict):
            continue
        test = ev.metrics.get("test", {}) if isinstance(ev.metrics, dict) else {}

        if e.pipeline.task_type == MLTaskType.CLASSIFICATION:
            v = test.get("accuracy", None)
            metric_name = "accuracy"
        elif e.pipeline.task_type == MLTaskType.REGRESSION:
            v = test.get("r2", None)
            metric_name = "r2"
        else:
            continue

        if v is None:
            continue

        labels.append(f"{e.algorithm.display_name} #{e.id}")
        values.append(float(v))

    if not values:
        return None

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.bar(range(len(values)), values)
    ax.set_title(f"Model Comparison ({metric_name})")
    ax.set_ylabel(metric_name)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    png = _fig_to_png_bytes(fig)
    return _upsert_plot(
        owner=owner,
        project=project,
        dataset_version=None,
        experiment=None,
        plot_type="project_model_comparison",
        title=f"Project Model Comparison ({metric_name})",
        metadata={"metric": metric_name, "count": len(values)},
        png_bytes=png,
    )
