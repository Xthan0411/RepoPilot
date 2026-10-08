"""Extract the reviewed P7 subset from downloaded SWE-bench rows API responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from repopilot.evals.dataset import SweBenchSourceRow

SELECTED_INSTANCE_IDS = (
    "django__django-15902",
    "mwaskom__seaborn-3190",
    "pallets__flask-4992",
    "pydata__xarray-4094",
    "pydata__xarray-5131",
    "pytest-dev__pytest-11143",
    "pytest-dev__pytest-11148",
    "sympy__sympy-20590",
)


def main() -> None:
    """Validate source responses and write only the fixed reviewed rows."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    selected: dict[str, SweBenchSourceRow] = {}
    for response_path in arguments.rows:
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if not isinstance(response, dict) or not isinstance(response.get("rows"), list):
            raise ValueError("rows API response is invalid")
        for wrapper in response["rows"]:
            if not isinstance(wrapper, dict) or not isinstance(wrapper.get("row"), dict):
                raise ValueError("rows API wrapper is invalid")
            raw_row = wrapper["row"]
            instance_id = raw_row.get("instance_id")
            if instance_id not in SELECTED_INSTANCE_IDS:
                continue
            row = SweBenchSourceRow.model_validate(raw_row)
            if row.instance_id in selected:
                raise ValueError("selected source row is duplicated")
            selected[row.instance_id] = row

    if set(selected) != set(SELECTED_INSTANCE_IDS):
        raise ValueError("selected source row is missing")
    output_rows = tuple(selected[item] for item in SELECTED_INSTANCE_IDS)
    TypeAdapter(tuple[SweBenchSourceRow, ...]).validate_python(output_rows)
    serialized = json.dumps(
        [item.model_dump(mode="json", by_alias=True) for item in output_rows],
        sort_keys=True,
        ensure_ascii=False,
        indent=2,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(f"{serialized}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
