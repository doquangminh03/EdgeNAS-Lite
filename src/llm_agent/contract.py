"""Validate structured responses from the requirement interpretation layer."""

from copy import deepcopy
from typing import Any, Dict, List

from src.requirement_parser.parser import parse_requirement_data


SUPPORTED_STATUSES = {
    "ready",
    "needs_clarification",
    "unsupported",
}

REQUIRED_KEYS = {
    "status",
    "requirement",
    "questions",
    "reasons",
}


class InterpretationValidationError(ValueError):
    """Raised when an interpretation violates the output contract."""


def _validate_text_list(value: Any, field: str) -> List[str]:
    if not isinstance(value, list):
        raise InterpretationValidationError(
            f"{field} must be a list."
        )

    result = []

    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise InterpretationValidationError(
                f"{field}[{index}] must be a non-empty string."
            )

        result.append(item.strip())

    return result


def validate_interpretation(data: Any) -> Dict[str, Any]:
    """Validate a decoded JSON response and return an independent copy."""

    if not isinstance(data, dict):
        raise InterpretationValidationError(
            "Interpretation must be a JSON object."
        )

    actual_keys = set(data)
    missing_keys = REQUIRED_KEYS - actual_keys
    unknown_keys = actual_keys - REQUIRED_KEYS

    if missing_keys:
        raise InterpretationValidationError(
            f"Missing fields: {sorted(missing_keys)}"
        )

    if unknown_keys:
        raise InterpretationValidationError(
            f"Unknown fields: {sorted(map(str, unknown_keys))}"
        )

    status = data["status"]

    if not isinstance(status, str) or status not in SUPPORTED_STATUSES:
        raise InterpretationValidationError(
            "status must be ready, needs_clarification, or unsupported."
        )

    questions = _validate_text_list(data["questions"], "questions")
    reasons = _validate_text_list(data["reasons"], "reasons")

    requirement = None

    if status == "ready":
        if questions or reasons:
            raise InterpretationValidationError(
                "A ready response must have empty questions and reasons."
            )

        if not isinstance(data["requirement"], dict):
            raise InterpretationValidationError(
                "A ready response must contain a requirement object."
            )

        # Reuse the project's existing requirement validation rules.
        # Work on a copy to preserve the caller's input.
        try:
            parsed = parse_requirement_data(
                deepcopy(data["requirement"])
            )
        except ValueError as error:
            raise InterpretationValidationError(
                f"Invalid requirement: {error}"
            ) from error

        requirement = parsed.to_dict()

    else:
        if data["requirement"] is not None:
            raise InterpretationValidationError(
                "A non-ready response must have requirement set to null."
            )

        if status == "needs_clarification":
            if not questions:
                raise InterpretationValidationError(
                    "A needs_clarification response must contain questions."
                )

            if reasons:
                raise InterpretationValidationError(
                    "Use questions for clarification; reasons must be empty."
                )

        if status == "unsupported":
            if not reasons:
                raise InterpretationValidationError(
                    "An unsupported response must contain reasons."
                )

            if questions:
                raise InterpretationValidationError(
                    "An unsupported response must have empty questions."
                )

    return {
        "status": status,
        "requirement": requirement,
        "questions": questions,
        "reasons": reasons,
    }