from __future__ import annotations

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from rest_framework.exceptions import ValidationError as DRFValidationError


def validate_json_schema_or_raise(schema: dict, instance: dict, context_prefix: str = "data") -> None:
    """
    Validates instance against JSON schema and raises DRF ValidationError with clear messages.
    """
    try:
        validator = Draft202012Validator(schema)
    except Exception as e:
        # Schema is stored in DB; if it’s malformed, it’s a server-side issue
        raise DRFValidationError({context_prefix: f"Internal schema error: {str(e)}"})

    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)
    if not errors:
        return

    # Build readable errors, e.g. hyperparameters.C -> message
    detail = {}
    for err in errors:
        path = ".".join([context_prefix] + [str(p) for p in err.path]) if err.path else context_prefix
        # Keep the most helpful message
        detail[path] = err.message

    raise DRFValidationError(detail)
