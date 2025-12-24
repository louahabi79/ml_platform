from __future__ import annotations

import threading
import traceback
import sys
import sklearn
import pandas as pd

from django.db import transaction
from django.utils import timezone

from apps.ml_models.models import Experiment, ExperimentStatus
from apps.ml_models.services.pipeline_validation import validate_pipeline_or_raise
# This import is fine because it comes from a DIFFERENT file
from apps.ml_models.services.experiment_runner import train_evaluate_and_persist


def _runtime_meta() -> dict:
    return {
        "python": sys.version,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
    }


@transaction.atomic
def start_experiment_run(experiment: Experiment) -> Experiment:
    """
    Marks experiment as RUNNING and sets started_at.
    """
    experiment.status = ExperimentStatus.RUNNING
    experiment.started_at = timezone.now()
    experiment.error_message = ""
    experiment.run_logs = ""
    # Merge existing meta with new meta
    current_meta = experiment.runtime_meta or {}
    experiment.runtime_meta = {**current_meta, **_runtime_meta()}
    
    experiment.save(update_fields=["status", "started_at", "error_message", "run_logs", "runtime_meta"])
    return experiment


@transaction.atomic
def finish_experiment_success(experiment: Experiment, run_logs: str = "") -> None:
    experiment.status = ExperimentStatus.SUCCEEDED
    experiment.finished_at = timezone.now()
    experiment.run_logs = (experiment.run_logs or "") + run_logs
    experiment.save(update_fields=["status", "finished_at", "run_logs"])


@transaction.atomic
def finish_experiment_failure(experiment: Experiment, error_message: str, run_logs: str = "") -> None:
    experiment.status = ExperimentStatus.FAILED
    experiment.finished_at = timezone.now()
    experiment.error_message = error_message[:20000]
    experiment.run_logs = (experiment.run_logs or "") + run_logs
    experiment.save(update_fields=["status", "finished_at", "error_message", "run_logs"])


def run_experiment_sync(experiment_id: int) -> None:
    """
    Synchronous execution trigger.
    Calls the full training pipeline from experiment_runner.
    """
    # Re-fetch experiment to ensure fresh state
    exp = Experiment.objects.select_related("pipeline", "dataset_version", "algorithm").get(id=experiment_id)
    
    try:
        start_experiment_run(exp)

        # 1. Safety check: validate pipeline consistency
        validate_pipeline_or_raise(exp.pipeline)

        # 2. Run the heavy ML training (Phase 5 logic)
        result = train_evaluate_and_persist(exp)

        # 3. Handle result
        if result["ok"]:
            finish_experiment_success(exp, run_logs=result.get("logs", ""))
        else:
            finish_experiment_failure(
                exp, 
                error_message=result.get("error", "Unknown error"), 
                run_logs=result.get("logs", "") + "\n" + result.get("traceback", "")
            )

    except Exception as e:
        # Catch-all for crashes outside the runner (e.g. DB issues)
        tb = traceback.format_exc()
        finish_experiment_failure(exp, error_message=str(e), run_logs=f"\n[CRITICAL ERROR]\n{tb}\n")


def run_experiment_threaded(experiment_id: int) -> None:
    """
    Simple threading option to avoid blocking the HTTP request.
    """
    t = threading.Thread(target=run_experiment_sync, args=(experiment_id,), daemon=True)
    t.start()