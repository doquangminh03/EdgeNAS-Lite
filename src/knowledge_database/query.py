from copy import deepcopy
from typing import Any, Dict, List, Optional


def query_candidates(
    knowledge: Dict[str, Any],
    *,
    dataset: Optional[str] = None,
    model_family: Optional[str] = None,
    device: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return candidates matching every supplied filter."""

    filters = (
        ("accuracy", "dataset", dataset),
        ("model", "family", model_family),
        ("benchmark", "device", device),
    )

    for _, field, expected in filters:
        if expected is not None:
            if not isinstance(expected, str) or not expected.strip():
                raise ValueError(
                    f"Filter {field} must be a non-empty string."
                )

    matches = []

    for item in knowledge["candidates"]:
        record = item["record"]
        matched = True

        for section_name, field, expected in filters:
            if expected is None:
                continue

            section = record.get(section_name)
            actual = (
                section.get(field)
                if isinstance(section, dict)
                else None
            )

            if (
                not isinstance(actual, str)
                or actual.strip().casefold()
                != expected.strip().casefold()
            ):
                matched = False
                break

        if matched:
            matches.append(deepcopy(item))

    return sorted(
        matches,
        key=lambda item: item["record"]["candidate_id"],
    )