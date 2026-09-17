"""Parse and validate a JSON interpretation response."""

import json
import math
from typing import Any, Dict

from src.llm_agent.contract import (
    InterpretationValidationError,
    validate_interpretation,
)


def _reject_constant(value: str) -> None:
    """Reject non-standard JSON constants such as NaN and Infinity."""
    raise InterpretationValidationError(
        f"Non-standard JSON numeric value: {value}"
    )


def _parse_finite_float(value: str) -> float:
    """Prevent very large JSON numbers from becoming infinity."""
    number = float(value)

    if not math.isfinite(number):
        raise InterpretationValidationError(
            f"JSON number must be finite: {value}"
        )

    return number


def _reject_duplicate_keys(pairs: list) -> Dict[str, Any]:
    """Reject ambiguous objects instead of silently replacing values."""
    result = {}

    for key, value in pairs:
        if key in result:
            raise InterpretationValidationError(
                f"Duplicate JSON key: {key}"
            )

        result[key] = value

    return result


def parse_interpretation_response(response_text: str) -> Dict[str, Any]:
    """Parse one complete JSON object and validate its contract."""

    if not isinstance(response_text, str) or not response_text.strip():
        raise InterpretationValidationError(
            "Response must be a non-empty string."
        )

    try:
        data = json.loads(
            response_text,
            parse_constant=_reject_constant,
            parse_float=_parse_finite_float,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise InterpretationValidationError(
            "Response must contain only valid JSON "
            f"(line {error.lineno}, column {error.colno})."
        ) from error

    return validate_interpretation(data)