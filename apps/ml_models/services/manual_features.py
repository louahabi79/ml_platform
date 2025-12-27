from __future__ import annotations

import ast
import math
from typing import Dict, Any, Iterable

import pandas as pd
from rest_framework.exceptions import ValidationError


_ALLOWED_FUNCS = {
    "log": math.log,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "abs": abs,
    "pow": pow,
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
    ast.Load,
    ast.Name,
    ast.Constant,
    ast.Call,
)


class _SafeExprValidator(ast.NodeVisitor):
    def __init__(self, allowed_names: set[str]):
        self.allowed_names = allowed_names

    def generic_visit(self, node):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValidationError({"manual_feature": f"Unsafe node type: {type(node).__name__}"})
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id not in self.allowed_names and node.id not in _ALLOWED_FUNCS:
            raise ValidationError({"manual_feature": f"Unknown name '{node.id}' in expression."})

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValidationError({"manual_feature": "Only functions allowed: log, sqrt, exp, abs, pow."})
        for arg in node.args:
            self.visit(arg)


def apply_manual_features_to_df(df: pd.DataFrame, features: Iterable[Dict[str, Any]]) -> pd.DataFrame:
    """
    features: iterable of {"name": str, "expression": str}
    Adds computed columns to df and returns a copy.
    Expressions operate on columns by name (vectorized via pandas Series ops).

    Example:
      name="log_income", expression="log(income + 1)"
    """
    df = df.copy()

    for f in features:
        name = (f.get("name") or "").strip()
        expr = (f.get("expression") or "").strip()

        if not name or not expr:
            raise ValidationError({"manual_features": "Each manual feature must have name and expression."})
        if name in df.columns:
            raise ValidationError({"manual_features": f"Feature name '{name}' already exists in dataset."})

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            raise ValidationError({"manual_features": f"Invalid expression '{expr}': {str(e)}"})

        allowed_names = set(df.columns) | set(_ALLOWED_FUNCS.keys())
        _SafeExprValidator(allowed_names=allowed_names).visit(tree)

        env = {col: df[col] for col in df.columns}
        env.update(_ALLOWED_FUNCS)

        try:
            result = eval(compile(tree, "<manual_feature>", "eval"), {"__builtins__": {}}, env)
        except Exception as e:
            raise ValidationError({"manual_features": f"Expression failed for '{name}': {str(e)}"})

        df[name] = result

    return df
