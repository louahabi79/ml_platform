from __future__ import annotations

import json
from typing import Any, Dict

import pandas as pd
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.http import FileResponse, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods
from rest_framework.exceptions import ValidationError as DRFValidationError

from apps.datasets.models import Project, Dataset, DatasetVersion
from apps.ml_models.models import Algorithm, Pipeline, PipelineStep, Experiment, MLTaskType, ModelArtifact, EvaluationResult, ManualFeature
from apps.api.v1.serializers import PipelineStepSerializer, ExperimentSerializer, PipelineSerializer
from apps.ml_models.services.experiment_service import run_experiment_threaded
from apps.visualization.services import generate_dataset_plots, plot_project_model_comparison
from apps.visualization.models import PlotArtifact
from apps.ml_models.services.predict_service import predict_csv_for_experiment
from apps.ml_models.services.report_generator import generate_experiment_report_pdf


def _ensure_owner(obj, user):
    owner_id = getattr(obj, "owner_id", None)
    if owner_id is None:
        if hasattr(obj, "dataset") and getattr(obj.dataset, "owner_id", None) != user.id:
            raise PermissionDenied
        return
    if owner_id != user.id:
        raise PermissionDenied


def _plot_url(plot: PlotArtifact) -> str | None:
    if hasattr(plot, "image_file") and getattr(plot, "image_file", None):
        return plot.image_file.url
    # Fallbacks
    if hasattr(plot, "image") and getattr(plot, "image", None):
        return plot.image.url
    if hasattr(plot, "file") and getattr(plot, "file", None):
        return plot.file.url
    return None


@login_required
def dashboard(request: HttpRequest) -> HttpResponse:
    projects = Project.objects.filter(owner=request.user).order_by("-updated_at")[:10]
    recent_datasets = Dataset.objects.filter(owner=request.user).order_by("-updated_at")[:10]
    recent_experiments = Experiment.objects.filter(owner=request.user).order_by("-created_at")[:10]
    return render(request, "dashboard.html", {
        "projects": projects,
        "recent_datasets": recent_datasets,
        "recent_experiments": recent_experiments,
    })


@login_required
def project_detail(request: HttpRequest, project_id: int) -> HttpResponse:
    project = get_object_or_404(Project, id=project_id, owner=request.user)
    datasets = project.datasets.order_by("-updated_at")
    pipelines = project.pipelines.order_by("-updated_at")
    experiments = project.experiments.order_by("-created_at")[:30]
    return render(request, "projects/detail.html", {
        "project": project,
        "datasets": datasets,
        "pipelines": pipelines,
        "experiments": experiments,
    })


@login_required
def project_comparison(request: HttpRequest, project_id: int) -> HttpResponse:
    """
    Renders a model comparison bar chart and a metrics table for all experiments in a project.
    """
    project = get_object_or_404(Project, id=project_id, owner=request.user)

    # Generate/update comparison plot (Accuracy for classification, R2 for regression)
    plot_obj = plot_project_model_comparison(project=project, owner=request.user)

    plot_url = _plot_url(plot_obj) if plot_obj else None

    experiments = (
        Experiment.objects.filter(project=project, owner=request.user)
        .select_related("algorithm", "pipeline")
        .order_by("-created_at")
    )

    rows = []
    for e in experiments:
        ev = EvaluationResult.objects.filter(experiment=e).first()
        metrics = (ev.metrics if ev and isinstance(ev.metrics, dict) else {})
        test = metrics.get("test", {}) if isinstance(metrics, dict) else {}

        primary_metric = None
        if e.pipeline.task_type == MLTaskType.CLASSIFICATION:
            primary_metric = test.get("accuracy")
        elif e.pipeline.task_type == MLTaskType.REGRESSION:
            primary_metric = test.get("r2")

        rows.append({
            "id": e.id,
            "name": e.name,
            "algorithm": e.algorithm.display_name,
            "task": e.pipeline.task_type,
            "status": e.status,
            "accuracy": test.get("accuracy"),
            "f1_macro": test.get("f1_macro"),
            "precision_macro": test.get("precision_macro"),
            "recall_macro": test.get("recall_macro"),
            "rmse": test.get("rmse"),
            "mae": test.get("mae"),
            "r2": test.get("r2"),
            "primary": primary_metric,
        })

    return render(request, "projects/comparison.html", {
        "project": project,
        "plot_url": plot_url,
        "rows": rows,
    })


@login_required
@require_http_methods(["GET", "POST"])
def dataset_upload(request: HttpRequest) -> HttpResponse:
    projects = Project.objects.filter(owner=request.user).order_by("-updated_at")

    if request.method == "GET":
        return render(request, "datasets/upload.html", {"projects": projects})

    project_id = request.POST.get("project_id")
    dataset_name = (request.POST.get("dataset_name") or "").strip()
    description = (request.POST.get("description") or "").strip()
    file = request.FILES.get("file")

    errors = {}
    if not project_id:
        errors["project_id"] = "Project is required."
    if not dataset_name:
        errors["dataset_name"] = "Dataset name is required."
    if not file:
        errors["file"] = "CSV file is required."

    # 20MB hard limit (also enforced in settings)
    MAX_CSV_MB = 20
    if file and file.size > MAX_CSV_MB * 1024 * 1024:
        errors["file"] = f"CSV too large (max {MAX_CSV_MB}MB)."

    if errors:
        return render(request, "datasets/upload.html", {"projects": projects, "errors": errors})

    project = get_object_or_404(Project, id=int(project_id), owner=request.user)

    dataset, _ = Dataset.objects.get_or_create(
        owner=request.user,
        project=project,
        name=dataset_name,
        defaults={
            "description": description,
            "source_type": "CSV_UPLOAD",
            "external_source": {},
        }
    )
    if description and dataset.description != description:
        dataset.description = description
        dataset.save(update_fields=["description", "updated_at"])

    content = file.read()
    sha256 = DatasetVersion.compute_sha256(content)
    latest = DatasetVersion.objects.filter(dataset=dataset).order_by("-version_number").first()
    next_version = (latest.version_number + 1) if latest else 1
    file.seek(0)

    dv = DatasetVersion.objects.create(
        dataset=dataset,
        created_by=request.user,
        version_number=next_version,
        file=file,
        sha256=sha256,
        row_count=0,
        column_count=0,
        schema_json={},
        profile_json={},
        notes="",
    )

    # Profile now (real)
    df = pd.read_csv(dv.file.path)
    dv.row_count = int(df.shape[0])
    dv.column_count = int(df.shape[1])
    dv.schema_json = {"columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]}
    dv.profile_json = {
        "missing_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "unique_by_column": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
    }
    dv.save(update_fields=["row_count", "column_count", "schema_json", "profile_json"])

    # Generate dataset plots (PNG)
    generate_dataset_plots(dataset_version=dv, df=df, owner=request.user, project=project)

    return redirect("dataset_version_detail", version_id=dv.id)


@login_required
def dataset_detail(request: HttpRequest, dataset_id: int) -> HttpResponse:
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    versions = dataset.versions.order_by("-version_number")
    return render(request, "datasets/detail.html", {"dataset": dataset, "versions": versions})


@login_required
def dataset_version_detail(request: HttpRequest, version_id: int) -> HttpResponse:
    """
    FIXED:
    - Builds row_data = [{'name','dtype','missing','unique'}, ...]
    - Fetches dataset plots (distributions + correlation heatmap) and passes URLs to template.
    """
    dv = get_object_or_404(DatasetVersion, id=version_id)
    _ensure_owner(dv, request.user)

    columns = (dv.schema_json or {}).get("columns", [])
    profile = dv.profile_json or {}
    missing_by = profile.get("missing_by_column", {}) or {}
    unique_by = profile.get("unique_by_column", {}) or {}

    row_data = []
    for c in columns:
        name = c.get("name")
        dtype = c.get("dtype")
        row_data.append({
            "name": name,
            "dtype": dtype,
            "missing": int(missing_by.get(name, 0)),
            "unique": int(unique_by.get(name, 0)),
        })

    # Fetch dataset-level plots (experiment is null)
    plots = PlotArtifact.objects.filter(dataset_version=dv, experiment__isnull=True)
    dv_plot_map: Dict[str, str] = {}
    for p in plots:
        url = _plot_url(p)
        if url:
            dv_plot_map[p.plot_type] = url

    return render(request, "datasets/version_detail.html", {
        "dv": dv,
        "row_data": row_data,
        "dv_plot_map": dv_plot_map,
    })


@login_required
@require_http_methods(["GET", "POST"])
def pipeline_wizard(request: HttpRequest, version_id: int) -> HttpResponse:
    dv = get_object_or_404(DatasetVersion, id=version_id)
    _ensure_owner(dv, request.user)

    dataset = dv.dataset
    project = dataset.project
    _ensure_owner(project, request.user)

    cols = [c["name"] for c in (dv.schema_json or {}).get("columns", [])]
    if not cols:
        df = pd.read_csv(dv.file.path, nrows=5)
        cols = list(df.columns)

    task_types = [
        {"value": MLTaskType.REGRESSION, "label": "Regression"},
        {"value": MLTaskType.CLASSIFICATION, "label": "Classification"},
        {"value": MLTaskType.CLUSTERING, "label": "Clustering"},
    ]

    if request.method == "GET":
        return render(request, "pipelines/wizard.html", {
            "project": project,
            "dataset": dataset,
            "dv": dv,
            "columns": cols,
            "task_types": task_types,
        })

    name = (request.POST.get("pipeline_name") or "").strip()
    task_type = request.POST.get("task_type") or ""
    target_column = (request.POST.get("target_column") or "").strip()
    steps_json = request.POST.get("steps_json") or "[]"
    feature_columns_json = request.POST.get("feature_columns_json") or "[]"
    manual_features_json = request.POST.get("manual_features_json") or "[]"

    errors = {}
    if not name:
        errors["pipeline_name"] = "Pipeline name is required."
    if task_type not in (MLTaskType.REGRESSION, MLTaskType.CLASSIFICATION, MLTaskType.CLUSTERING):
        errors["task_type"] = "Valid task type is required."
    if task_type in (MLTaskType.REGRESSION, MLTaskType.CLASSIFICATION) and (not target_column or target_column not in cols):
        errors["target_column"] = "Target column is required for supervised tasks."

    try:
        steps_payload = json.loads(steps_json)
        if not isinstance(steps_payload, list):
            raise ValueError
    except Exception:
        errors["steps_json"] = "Invalid steps JSON."

    try:
        feature_cols = json.loads(feature_columns_json)
        if not isinstance(feature_cols, list):
            raise ValueError
    except Exception:
        errors["feature_columns_json"] = "Invalid feature columns JSON."
        feature_cols = []

    try:
        manual_defs = json.loads(manual_features_json)
        if not isinstance(manual_defs, list):
            raise ValueError
    except Exception:
        errors["manual_features_json"] = "Invalid manual features JSON."
        manual_defs = []

    # Ensure target column isn't included as a feature for supervised
    if task_type in (MLTaskType.REGRESSION, MLTaskType.CLASSIFICATION) and target_column:
        feature_cols = [c for c in feature_cols if c != target_column]

    if errors:
        return render(request, "pipelines/wizard.html", {
            "project": project,
            "dataset": dataset,
            "dv": dv,
            "columns": cols,
            "task_types": task_types,
            "errors": errors,
        })

    pipeline_data = {
        "project": project.id,
        "dataset_version": dv.id,
        "name": name,
        "task_type": task_type,
        "target_column": target_column if task_type != MLTaskType.CLUSTERING else "",
        "feature_columns": feature_cols,  # now supported by UI
        "random_seed": 42,
    }
    pser = PipelineSerializer(data=pipeline_data)
    pser.is_valid(raise_exception=True)
    pipeline = pser.save(owner=request.user)

    for idx, step in enumerate(steps_payload):
        sdata = {
            "pipeline": pipeline.id,
            "step_type": step.get("step_type"),
            "order": idx,
            "enabled": bool(step.get("enabled", True)),
            "config": step.get("config") or {},
        }
        sser = PipelineStepSerializer(data=sdata)
        sser.is_valid(raise_exception=True)
        sser.save()

    # Save manual features (Safe evaluation happens in service during training)
    for mf in manual_defs:
        n = (mf.get("name") or "").strip()
        expr = (mf.get("expression") or "").strip()
        if n and expr:
            ManualFeature.objects.create(pipeline=pipeline, name=n, expression=expr)

    return redirect("pipeline_detail", pipeline_id=pipeline.id)


@login_required
def pipeline_detail(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    pipeline = get_object_or_404(Pipeline, id=pipeline_id, owner=request.user)
    steps = PipelineStep.objects.filter(pipeline=pipeline).order_by("order")
    mfs = ManualFeature.objects.filter(pipeline=pipeline).order_by("created_at") if hasattr(ManualFeature, "created_at") else ManualFeature.objects.filter(pipeline=pipeline)
    return render(request, "pipelines/detail.html", {"pipeline": pipeline, "steps": steps, "manual_features": mfs})


@login_required
@require_http_methods(["GET", "POST"])
def experiment_create(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    pipeline = get_object_or_404(Pipeline, id=pipeline_id, owner=request.user)
    algorithms = Algorithm.objects.all().order_by("family", "display_name")

    if request.method == "GET":
        algo_payload = []
        for a in algorithms:
            algo_payload.append({
                "id": a.id,
                "key": a.key,
                "display_name": a.display_name,
                "family": a.family,
                "supported_tasks": a.supported_tasks,
                "hyperparameter_schema": a.hyperparameter_schema,
                "default_hyperparameters": a.default_hyperparameters,
            })
        return render(request, "experiments/create.html", {
            "pipeline": pipeline,
            "algorithms": algorithms,
            "algorithms_json": json.dumps(algo_payload),
            "task_type": pipeline.task_type,
        })

    name = (request.POST.get("name") or "").strip()
    algorithm_id = request.POST.get("algorithm_id")
    use_cv = request.POST.get("use_cross_validation") == "on"
    cv_folds = int(request.POST.get("cv_folds") or 5)
    test_size = float(request.POST.get("test_size") or 0.2)
    hyperparams_json = request.POST.get("hyperparameters_json") or "{}"

    errors = {}
    if not name:
        errors["name"] = "Experiment name is required."
    if not algorithm_id:
        errors["algorithm_id"] = "Algorithm is required."

    try:
        hyperparams = json.loads(hyperparams_json)
        if not isinstance(hyperparams, dict):
            raise ValueError
    except Exception:
        errors["hyperparameters_json"] = "Invalid hyperparameters JSON."
        hyperparams = {}

    if errors:
        algo_payload = []
        for a in algorithms:
            algo_payload.append({
                "id": a.id,
                "key": a.key,
                "display_name": a.display_name,
                "family": a.family,
                "supported_tasks": a.supported_tasks,
                "hyperparameter_schema": a.hyperparameter_schema,
                "default_hyperparameters": a.default_hyperparameters,
            })
        return render(request, "experiments/create.html", {
            "pipeline": pipeline,
            "algorithms": algorithms,
            "algorithms_json": json.dumps(algo_payload),
            "task_type": pipeline.task_type,
            "errors": errors,
        })

    algo = get_object_or_404(Algorithm, id=int(algorithm_id))

    exp_data = {
        "project": pipeline.project_id,
        "dataset_version": pipeline.dataset_version_id,
        "pipeline": pipeline.id,
        "algorithm": algo.id,
        "name": name,
        "notes": "",
        "hyperparameters": hyperparams,
        "test_size": test_size,
        "use_cross_validation": use_cv,
        "cv_folds": cv_folds,
        "random_seed": pipeline.random_seed,
    }
    ser = ExperimentSerializer(data=exp_data)
    ser.is_valid(raise_exception=True)
    exp = ser.save(owner=request.user)

    run_experiment_threaded(exp.id)
    return redirect("experiment_results", experiment_id=exp.id)


@login_required
def experiment_results(request: HttpRequest, experiment_id: int) -> HttpResponse:
    exp = get_object_or_404(
        Experiment.objects.select_related("algorithm", "pipeline", "project"),
        id=experiment_id,
        owner=request.user,
    )
    evaluation = EvaluationResult.objects.filter(experiment=exp).first()
    artifact = ModelArtifact.objects.filter(experiment=exp).first()

    metrics = evaluation.metrics if evaluation else {}
    confusion = evaluation.confusion_matrix if evaluation else {}

    test_metrics = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    cv_metrics = metrics.get("cv", {}) if isinstance(metrics, dict) else {}

    cm_matrix = (confusion or {}).get("matrix", [])

    plots = PlotArtifact.objects.filter(experiment=exp).order_by("created_at")
    plot_map: Dict[str, str] = {}
    for p in plots:
        url = _plot_url(p)
        if url:
            plot_map[p.plot_type] = url

    return render(request, "experiments/results.html", {
        "exp": exp,
        "evaluation": evaluation,
        "artifact": artifact,
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
        "cm_matrix": cm_matrix,
        "plot_map": plot_map,
    })


@login_required
def download_model(request: HttpRequest, experiment_id: int) -> HttpResponse:
    exp = get_object_or_404(Experiment, id=experiment_id, owner=request.user)
    artifact = get_object_or_404(ModelArtifact, experiment=exp)
    return FileResponse(artifact.file.open("rb"), as_attachment=True, filename=artifact.file.name.split("/")[-1])


@login_required
@require_http_methods(["GET", "POST"])
def predict_view(request: HttpRequest, experiment_id: int) -> HttpResponse:
    exp = get_object_or_404(Experiment.objects.select_related("project", "algorithm"), id=experiment_id, owner=request.user)

    if request.method == "GET":
        return render(request, "experiments/predict.html", {"exp": exp})

    file = request.FILES.get("file")
    if not file:
        return render(request, "experiments/predict.html", {"exp": exp, "errors": {"file": "CSV file is required."}})

    try:
        csv_bytes = predict_csv_for_experiment(exp, file)
    except DRFValidationError as e:
        return render(request, "experiments/predict.html", {
            "exp": exp,
            "errors": e.detail if isinstance(e.detail, dict) else {"error": str(e)}
        })

    filename = f"predictions_experiment_{exp.id}.csv"
    resp = HttpResponse(csv_bytes, content_type="text/csv; charset=utf-8")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp


@login_required
def download_report_pdf(request: HttpRequest, experiment_id: int) -> HttpResponse:
    exp = get_object_or_404(Experiment.objects.select_related("project", "algorithm"), id=experiment_id, owner=request.user)

    pdf_bytes = generate_experiment_report_pdf(exp)
    filename = f"experiment_report_{exp.id}.pdf"

    resp = HttpResponse(pdf_bytes, content_type="application/pdf")
    resp["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp
