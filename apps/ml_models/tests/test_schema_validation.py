from __future__ import annotations

from django.test import TestCase
from rest_framework.exceptions import ValidationError

from apps.ml_models.services.schema_validation import validate_json_schema_or_raise


class SchemaValidationTests(TestCase):
    def test_invalid_hyperparameters_raise_validation_error(self):
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "n_estimators": {"type": "integer", "minimum": 1},
                "max_depth": {"type": ["integer", "null"], "minimum": 1},
            },
            "required": ["n_estimators"],
        }

        invalid_instance = {"n_estimators": "100"}  # wrong type: should be integer
        with self.assertRaises(ValidationError):
            validate_json_schema_or_raise(schema=schema, instance=invalid_instance, context_prefix="hyperparameters")
