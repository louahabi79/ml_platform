from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.http import FileResponse, HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_http_methods

from apps.datasets.models import Project, Dataset, DatasetVersion
from apps.ml_models.models import Algorithm, Pipeline, PipelineStep, Experiment, MLTaskType, ModelArtifact, EvaluationResult
from apps.api.v1.serializers import PipelineStepSerializer, ExperimentSerializer, PipelineSerializer
from apps.ml_models.services.experiment_service import run_experiment_threaded

from apps.visualization.services import generate_dataset_plots
from apps.visualization.models import PlotArtifact

from django.views.decorators.http import require_http_methods
from rest_framework.exceptions import ValidationError as DRFValidationError

from apps.ml_models.services.predict_service import predict_csv_for_experiment
from apps.ml_models.services.report_generator import generate_experiment_report_pdf



def _ensure_owner(obj, user):
    owner_id = getattr(obj, "owner_id", None)
    if owner_id is None:
        # dataset_version -> dataset -> owner
        if hasattr(obj, "dataset") and getattr(obj.dataset, "owner_id", None) != user.id:
            raise PermissionDenied
        return
    if owner_id != user.id:
        raise PermissionDenied


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

    # Basic profiling now (real, no placeholders)
    df = pd.read_csv(dv.file.path)
    dv.row_count = int(df.shape[0])
    dv.column_count = int(df.shape[1])
    dv.schema_json = {"columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]}
    dv.profile_json = {
        "missing_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
        "unique_by_column": {c: int(df[c].nunique(dropna=True)) for c in df.columns},
    }
    dv.save(update_fields=["row_count", "column_count", "schema_json", "profile_json"])
    generate_dataset_plots(dataset_version=dv, df=df, owner=request.user, project=project)
    return redirect("dataset_version_detail", version_id=dv.id)


@login_required
def dataset_detail(request: HttpRequest, dataset_id: int) -> HttpResponse:
    dataset = get_object_or_404(Dataset, id=dataset_id, owner=request.user)
    versions = dataset.versions.order_by("-version_number")
    return render(request, "datasets/detail.html", {"dataset": dataset, "versions": versions})


@login_required
def dataset_version_detail(request: HttpRequest, version_id: int) -> HttpResponse:
    dv = get_object_or_404(DatasetVersion, id=version_id)
    _ensure_owner(dv, request.user)

    columns = (dv.schema_json or {}).get("columns", [])
    profile = dv.profile_json or {}
    return render(request, "datasets/version_detail.html", {
        "dv": dv,
        "columns": columns,
        "profile": profile,
    })


@login_required
@require_http_methods(["GET", "POST"])
def pipeline_wizard(request: HttpRequest, version_id: int) -> HttpResponse:
    dv = get_object_or_404(DatasetVersion, id=version_id)
    _ensure_owner(dv, request.user)

    dataset = dv.dataset
    project = dataset.project
    _ensure_owner(project, request.user)

    # Columns list (prefer schema_json; fallback to reading csv)
    cols = [c["name"] for c in (dv.schema_json or {}).get("columns", [])]
    if not cols:
        df = pd.read_csv(dv.file.path, nrows=5)
        cols = list(df.columns)

    if request.method == "GET":
        return render(request, "pipelines/wizard.html", {
            "project": project,
            "dataset": dataset,
            "dv": dv,
            "columns": cols,
            "task_types": [
                {"value": MLTaskType.REGRESSION, "label": "Regression"},
                {"value": MLTaskType.CLASSIFICATION, "label": "Classification"},
                {"value": MLTaskType.CLUSTERING, "label": "Clustering"},
            ],
        })

    # POST: create Pipeline + steps from JSON payload
    name = (request.POST.get("pipeline_name") or "").strip()
    task_type = request.POST.get("task_type") or ""
    target_column = (request.POST.get("target_column") or "").strip()
    steps_json = request.POST.get("steps_json") or "[]"

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

    if errors:
        return render(request, "pipelines/wizard.html", {
            "project": project,
            "dataset": dataset,
            "dv": dv,
            "columns": cols,
            "task_types": [
                {"value": MLTaskType.REGRESSION, "label": "Regression"},
                {"value": MLTaskType.CLASSIFICATION, "label": "Classification"},
                {"value": MLTaskType.CLUSTERING, "label": "Clustering"},
            ],
            "errors": errors,
        })

    # Create pipeline
    pipeline_data = {
        "project": project.id,
        "dataset_version": dv.id,
        "name": name,
        "task_type": task_type,
        "target_column": target_column if task_type != MLTaskType.CLUSTERING else "",
        "feature_columns": [],  # Phase 6 wizard uses default: all except target
        "random_seed": 42,
    }
    pser = PipelineSerializer(data=pipeline_data)
    pser.is_valid(raise_exception=True)
    pipeline = pser.save(owner=request.user)

    # Create steps via serializer (polymorphic validation)
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

    return redirect("pipeline_detail", pipeline_id=pipeline.id)


@login_required
def pipeline_detail(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    pipeline = get_object_or_404(Pipeline, id=pipeline_id, owner=request.user)
    steps = PipelineStep.objects.filter(pipeline=pipeline).order_by("order")
    return render(request, "pipelines/detail.html", {"pipeline": pipeline, "steps": steps})


@login_required
@require_http_methods(["GET", "POST"])
def experiment_create(request: HttpRequest, pipeline_id: int) -> HttpResponse:
    pipeline = get_object_or_404(Pipeline, id=pipeline_id, owner=request.user)
    algorithms = Algorithm.objects.all().order_by("family", "display_name")

    if request.method == "GET":
        # Provide schemas to JS
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

    # POST
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

    if errors:
        # re-render with schemas
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

    # MVP strategy: threaded so SSR doesn’t block
    run_experiment_threaded(exp.id)

    return redirect("experiment_results", experiment_id=exp.id)


@login_required
def experiment_results(request: HttpRequest, experiment_id: int) -> HttpResponse:
    exp = get_object_or_404(Experiment.objects.select_related("algorithm", "pipeline"), id=experiment_id, owner=request.user)
    evaluation = EvaluationResult.objects.filter(experiment=exp).first()
    artifact = ModelArtifact.objects.filter(experiment=exp).first()

    metrics = evaluation.metrics if evaluation else {}
    confusion = evaluation.confusion_matrix if evaluation else {}

    # Flatten metrics for table rendering
    test_metrics = metrics.get("test", {}) if isinstance(metrics, dict) else {}
    cv_metrics = metrics.get("cv", {}) if isinstance(metrics, dict) else {}

    cm_matrix = (confusion or {}).get("matrix", [])
    
    # --- VISUALIZATION LOGIC ---
    plots = PlotArtifact.objects.filter(experiment=exp).order_by("created_at")
    plot_map = {}
    for p in plots:
        # Support either p.image or p.file (our model uses image_file field actually)
        if hasattr(p, "image_file") and p.image_file:
             plot_map[p.plot_type] = p.image_file.url
        elif hasattr(p, "image") and p.image:
            plot_map[p.plot_type] = p.image.url
        elif hasattr(p, "file") and p.file:
            plot_map[p.plot_type] = p.file.url

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
    exp = get_object_or_404(Experiment.objects.select_related("project"), id=experiment_id, owner=request.user)

    if request.method == "GET":
        return render(request, "experiments/predict.html", {"exp": exp})

    file = request.FILES.get("file")
    if not file:
        return render(request, "experiments/predict.html", {
            "exp": exp,
            "errors": {"file": "CSV file is required."}
        })

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
