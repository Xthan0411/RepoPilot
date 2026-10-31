"""Rebuild S7 outputs offline. Does not execute any evaluation task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repopilot.evals.report_formats import render_markdown, render_slots_csv
from repopilot.evals.reports import build_v6_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true", help="Compare saved outputs; write nothing")
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = args.output_dir or root / "results/p7-holdout"
    report = build_v6_report(repository_root=root, evidence_root=args.evidence_root)
    outputs = {
        "aggregate-v6.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        "aggregate-v6-slots.csv": render_slots_csv(report),
        "aggregate-v6.md": render_markdown(report),
    }
    if args.check:
        for name, content in outputs.items():
            path = output / name
            if not path.is_file() or path.read_bytes() != content.encode():
                raise SystemExit(f"aggregate_mismatch:{name}")
    else:
        # Derived files can be recreated, but never replace differing prior content
        # implicitly. Choose a new output directory after a deliberate code change.
        for name, content in outputs.items():
            path = output / name
            if path.exists() and (path.is_symlink() or path.read_bytes() != content.encode()):
                raise SystemExit(f"output_exists_with_different_content:{name}")
        output.mkdir(parents=True, exist_ok=True)
        for name, content in outputs.items():
            path = output / name
            if not path.exists():
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(content)
    print(
        json.dumps(
            {
                "aggregate_verified": True,
                "slots": report["planned_slots"],
                "content_sha256": report["content_sha256"],
                "check_only": args.check,
                "provider_requests": 0,
                "sandbox_test_runs": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
