from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
from django.db.models import QuerySet
from rest_framework.exceptions import ValidationError

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold

from apps.ml_models.models import Pipeline, PipelineStep, PipelineStepType, MLTaskType


@dataclass(frozen=True)
class BuiltPipeline:
    sklearn_pipeline: SkPipeline
    X: pd.DataFrame
    y: Optional[pd.Series]
    numeric_cols: List[str]
    categorical_cols: List[str]


def _infer_column_types(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    sub = df[feature_cols]
    numeric = list(sub.select_dtypes(include=["number", "bool"]).columns)
    categorical = [c for c in feature_cols if c not in numeric]
    return numeric, categorical


def _get_steps(pipeline_obj: Pipeline) -> QuerySet[PipelineStep]:
    return PipelineStep.objects.filter(pipeline=pipeline_obj, enabled=True).order_by("order", "id")


def _build_imputer_configs(step: PipelineStep) -> Tuple[Optional[SimpleImputer], Optional[SimpleImputer], dict]:
    """
    Returns (numeric_imputer, categorical_imputer, df_level_ops)
    df_level_ops handles drop_rows/drop_columns strategies.
    """
    cfg = step.config or {}
    strategy = cfg.get("strategy")
    fill_value = cfg.get("fill_value", None)
    cols = cfg.get("columns") or []
    df_ops = {"drop_rows": False, "drop_columns": []}  # columns to drop

    if strategy in ("drop_rows", "drop_columns"):
        if strategy == "drop_rows":
            df_ops["drop_rows"] = True
        else:
            # if columns specified, drop them; else drop any column with missing values (applied later)
            df_ops["drop_columns"] = cols
        return None, None, df_ops

    # Numeric imputer
    if strategy in ("mean", "median"):
        num_imp = SimpleImputer(strategy=strategy)
        # Categorical can't do mean/median → fallback
        cat_strategy = "most_frequent"
        cat_imp = SimpleImputer(strategy=cat_strategy)
    elif strategy == "most_frequent":
        num_imp = SimpleImputer(strategy="most_frequent")
        cat_imp = SimpleImputer(strategy="most_frequent")
    elif strategy == "constant":
        num_imp = SimpleImputer(strategy="constant", fill_value=fill_value)
        cat_imp = SimpleImputer(strategy="constant", fill_value=fill_value)
    else:
        raise ValidationError({"pipeline.missing_values.strategy": f"Unsupported missing strategy: {strategy}"})

    return num_imp, cat_imp, df_ops


def _build_encoder(step: PipelineStep):
    cfg = step.config or {}
    enc = cfg.get("encoder")
    handle_unknown = cfg.get("handle_unknown", "ignore")

    if enc == "onehot":
        # sklearn >= 1.2 uses sparse_output, not sparse
        return OneHotEncoder(
            handle_unknown=handle_unknown,
            sparse_output=False,
        )
    if enc == "ordinal":
        # For unknown categories, encode as -1
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    raise ValidationError({"pipeline.encoding.encoder": f"Unsupported encoder: {enc}"})


def _build_scaler(step: PipelineStep):
    cfg = step.config or {}
    scaler = cfg.get("scaler")

    if scaler == "none":
        return None
    if scaler == "standard":
        return StandardScaler()
    if scaler == "minmax":
        return MinMaxScaler()
    if scaler == "robust":
        return RobustScaler()
    raise ValidationError({"pipeline.scaling.scaler": f"Unsupported scaler: {scaler}"})


def _build_polynomial(step: PipelineStep):
    cfg = step.config or {}
    degree = int(cfg.get("degree", 2))
    include_bias = bool(cfg.get("include_bias", False))
    interaction_only = bool(cfg.get("interaction_only", False))
    return PolynomialFeatures(degree=degree, include_bias=include_bias, interaction_only=interaction_only)


def build_sklearn_pipeline(
    pipeline_obj: Pipeline,
    df: pd.DataFrame,
    estimator,
) -> BuiltPipeline:
    """
    Converts stored PipelineSteps into a fitted-ready sklearn Pipeline:
      (preprocess ColumnTransformer) -> (optional feature selection) -> estimator

    Notes:
    - Encoding/scaling/polynomial apply by type (numeric vs categorical).
    - Missing values drop_rows/drop_columns are handled at DataFrame level before sklearn transformers.
    """
    if pipeline_obj.task_type in (MLTaskType.REGRESSION, MLTaskType.CLASSIFICATION) and not pipeline_obj.target_column:
        raise ValidationError({"pipeline.target_column": "Target column is required for supervised tasks."})

    # Feature columns: empty means "all except target"
    if pipeline_obj.feature_columns:
        feature_cols = list(pipeline_obj.feature_columns)
    else:
        feature_cols = [c for c in df.columns if c != pipeline_obj.target_column]

    if not feature_cols:
        raise ValidationError({"pipeline.feature_columns": "No feature columns selected."})

    # Split X/y
    y = None
    if pipeline_obj.task_type in (MLTaskType.REGRESSION, MLTaskType.CLASSIFICATION):
        if pipeline_obj.target_column not in df.columns:
            raise ValidationError({"pipeline.target_column": f"Target column '{pipeline_obj.target_column}' not found in dataset."})
        y = df[pipeline_obj.target_column]
    X = df[feature_cols].copy()

    numeric_cols, categorical_cols = _infer_column_types(df, feature_cols)

    # Build per-type transformer lists, driven by ordered steps
    numeric_steps = []
    categorical_steps = []
    df_ops = {"drop_rows": False, "drop_columns": []}
    use_feature_selection = None  # currently VarianceThreshold only (real, deterministic)
    feature_selection_cfg = None

    steps = list(_get_steps(pipeline_obj))
    for s in steps:
        if s.step_type == PipelineStepType.MISSING_VALUES:
            num_imp, cat_imp, ops = _build_imputer_configs(s)
            df_ops["drop_rows"] = df_ops["drop_rows"] or ops.get("drop_rows", False)
            if ops.get("drop_columns"):
                df_ops["drop_columns"].extend(list(ops["drop_columns"]))

            if num_imp is not None:
                numeric_steps.append(("imputer", num_imp))
            if cat_imp is not None:
                categorical_steps.append(("imputer", cat_imp))

        elif s.step_type == PipelineStepType.ENCODING:
            enc = _build_encoder(s)
            categorical_steps.append(("encoder", enc))

        elif s.step_type == PipelineStepType.SCALING:
            sc = _build_scaler(s)
            if sc is not None:
                numeric_steps.append(("scaler", sc))

        elif s.step_type == PipelineStepType.POLYNOMIAL_FEATURES:
            poly = _build_polynomial(s)
            numeric_steps.append(("poly", poly))

        elif s.step_type == PipelineStepType.FEATURE_SELECTION:
            # Real minimal implementation: VarianceThreshold / none.
            cfg = s.config or {}
            method = cfg.get("method", "none")
            if method == "none":
                continue
            if method == "variance_threshold":
                thr = cfg.get("threshold", None)
                if thr is None:
                    raise ValidationError({"pipeline.feature_selection.threshold": "Required for variance_threshold."})
                use_feature_selection = "variance_threshold"
                feature_selection_cfg = {"threshold": float(thr)}
            else:
                # k_best/model_based supported later in a dedicated phase expansion
                raise ValidationError({"pipeline.feature_selection.method": f"Not yet supported in Phase 5 runner: {method}"})

        elif s.step_type == PipelineStepType.MANUAL_FEATURES:
            # Manual features are applied later in Phase 6 (safe expression parsing).
            # For Phase 5 runner we do not silently ignore: we enforce that there are no manual features yet.
            # This keeps behavior academically honest and prevents wrong results.
            if pipeline_obj.manual_features.exists():
                raise ValidationError({"pipeline.manual_features": "Manual features exist but are not enabled in Phase 5 runner yet."})
            continue

    # Apply DF-level missing strategies
    if df_ops["drop_columns"]:
        drop_set = set([c for c in df_ops["drop_columns"] if c in X.columns])
        X = X.drop(columns=list(drop_set))
        numeric_cols = [c for c in numeric_cols if c not in drop_set]
        categorical_cols = [c for c in categorical_cols if c not in drop_set]

    if df_ops["drop_rows"]:
        # drop rows where any feature or target (if exists) is missing
        if y is not None:
            combined = pd.concat([X, y.rename("__target__")], axis=1)
            combined = combined.dropna(axis=0)
            y = combined["__target__"]
            X = combined.drop(columns=["__target__"])
        else:
            X = X.dropna(axis=0)

    if numeric_cols:
        num_pipe = SkPipeline(steps=numeric_steps) if numeric_steps else "passthrough"
    else:
        num_pipe = "drop"

    if categorical_cols:
        # If encoding is not configured but categorical columns exist, this will fail downstream.
        # We enforce explicit encoding for clarity.
        has_encoder = any(name == "encoder" for name, _ in categorical_steps)
        if not has_encoder:
            raise ValidationError({"pipeline.encoding": "Categorical columns detected but no ENCODING step configured."})
        cat_pipe = SkPipeline(steps=categorical_steps) if categorical_steps else "passthrough"
    else:
        cat_pipe = "drop"

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    full_steps = [("preprocess", preprocess)]

    if use_feature_selection == "variance_threshold":
        selector = VarianceThreshold(threshold=feature_selection_cfg["threshold"])
        full_steps.append(("feature_select", selector))

    full_steps.append(("model", estimator))

    sklearn_pipeline = SkPipeline(steps=full_steps)

    return BuiltPipeline(
        sklearn_pipeline=sklearn_pipeline,
        X=X,
        y=y,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )
