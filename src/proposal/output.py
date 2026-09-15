import json
from pathlib import Path
from typing import Any, Dict, Union


def save_proposal(
    proposal: Dict[str, Any],
    output_file: Union[str, Path],
) -> Path:
    """Save a proposal result as JSON and return its path."""

    output_path = Path(output_file)

    # Serialize before writing; reject NaN and infinity.
    content = json.dumps(
        proposal,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")

    return output_path