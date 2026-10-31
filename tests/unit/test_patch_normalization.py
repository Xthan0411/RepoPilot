"""Synthetic count normalization and exact-context offset safety regressions."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from repopilot.enums import PatchErrorCode
from repopilot.sandbox.patching import APPLY_PATCH_HELPER
from repopilot.tools import PatchLimits, PatchValidationError, parse_unified_diff

VALID_DIFF = (
    "diff --git a/pkg/module.py b/pkg/module.py\n"
    "--- a/pkg/module.py\n"
    "+++ b/pkg/module.py\n"
    "@@ -1,2 +1,2 @@\n"
    "-VALUE = 1\n"
    "+VALUE = 2\n"
    " KEEP = True\n"
)


def limits(**updates: int) -> PatchLimits:
    return PatchLimits.model_validate(
        {
            "max_bytes": 262_144,
            "max_files": 20,
            "max_hunks": 100,
            "max_changed_lines": 2_000,
            **updates,
        }
    )


def helper() -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    # Execute the shipped helper definitions, not its filesystem entry point.
    definitions = APPLY_PATCH_HELPER.split("\nname = sys.argv[1]\n", maxsplit=1)[0]
    exec(compile(definitions, "patch-helper-definitions", "exec"), namespace)
    return namespace


def diff(*hunks: str, path: str = "module.py") -> str:
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n" + "".join(hunks)


def apply(source: str, candidate: str) -> bytes:
    parsed = parse_unified_diff(candidate, limits())
    implementation = helper()
    files = implementation["parse_patch"](parsed.content, 20, 100, 2_000)
    result = implementation["apply_hunks"](source, files[0][1])
    assert isinstance(result, bytes)
    return result


def assert_conflict(source: str, candidate: str) -> None:
    implementation = helper()
    parsed = parse_unified_diff(candidate, limits())
    files = implementation["parse_patch"](parsed.content, 20, 100, 2_000)
    with pytest.raises(implementation["PatchFailure"]) as captured:
        implementation["apply_hunks"](source, files[0][1])
    assert captured.value.code == "apply_conflict"


@pytest.mark.parametrize("header", ["@@ -1,99 +1,0 @@", "@@ -1 +1 @@", "@@ -1,0 +1,99 @@"])
def test_only_hunk_counts_are_normalized_and_both_identities_are_recorded(header: str) -> None:
    proposed = VALID_DIFF.replace("@@ -1,2 +1,2 @@", header)
    parsed = parse_unified_diff(proposed, limits())
    assert parsed.content == VALID_DIFF
    assert parsed.encoded == VALID_DIFF.encode()
    assert parsed.summary.sha256 == hashlib.sha256(parsed.encoded).hexdigest()
    assert parsed.summary.original_sha256 == hashlib.sha256(proposed.encode()).hexdigest()
    assert parsed.summary.original_size_bytes == len(proposed.encode())
    assert parsed.summary.normalized_hunk_count == 1


def test_correct_diff_bytes_and_section_text_are_preserved() -> None:
    canonical = VALID_DIFF.replace(" @@\n", " @@ def example(value):\n")
    parsed = parse_unified_diff(canonical, limits())
    assert parsed.content == canonical
    assert parsed.summary.original_sha256 == parsed.summary.sha256
    assert parsed.summary.normalized_hunk_count == 0
    wrong = canonical.replace("-1,2 +1,2", "-1,30 +1,30")
    assert parse_unified_diff(wrong, limits()).content == canonical


@pytest.mark.parametrize("path", ["../escape.py", ".git/config", ".env", "/absolute.py"])
def test_wrong_counts_do_not_bypass_path_policy(path: str) -> None:
    with pytest.raises(PatchValidationError) as captured:
        parse_unified_diff(diff("@@ -1,99 +1,99 @@\n-a\n+b\n", path=path), limits())
    assert captured.value.code is PatchErrorCode.PATH_DENIED


def test_declared_small_counts_do_not_hide_actual_hunk_overlap() -> None:
    candidate = diff(
        "@@ -1,1 +1,1 @@\n context\n-old\n+new\n last\n",
        "@@ -3,1 +3,1 @@\n-last\n+other\n",
    )
    with pytest.raises(PatchValidationError) as captured:
        parse_unified_diff(candidate, limits())
    assert captured.value.code is PatchErrorCode.INVALID_DIFF


@pytest.mark.parametrize(
    "hunk",
    [
        "@@ -0,0 +1,1 @@\n-old\n+new\n",
        "@@ -1,1 +0,0 @@\n-old\n+new\n",
        "@@ -1,99 +1,99 @@\n unchanged\n",
        "@@ -1,99 +1,99 @@\n-old\n+new\n\\ No newline at end of file\n",
    ],
)
def test_semantically_invalid_hunks_are_not_repaired(hunk: str) -> None:
    with pytest.raises(PatchValidationError):
        parse_unified_diff(diff(hunk), limits())


def test_original_and_canonical_byte_limits_are_both_enforced() -> None:
    huge_header = VALID_DIFF.replace("-1,2", "-1,999999")
    with pytest.raises(PatchValidationError) as original:
        parse_unified_diff(huge_header, limits(max_bytes=len(VALID_DIFF.encode())))
    assert original.value.code is PatchErrorCode.PATCH_TOO_LARGE
    omitted_counts = diff("@@ -1 +1 @@\n-a\n+b\n context\n")
    with pytest.raises(PatchValidationError) as normalized:
        parse_unified_diff(omitted_counts, limits(max_bytes=len(omitted_counts.encode())))
    assert normalized.value.code is PatchErrorCode.PATCH_TOO_LARGE


def test_wrong_counts_do_not_bypass_changed_line_limit() -> None:
    with pytest.raises(PatchValidationError) as captured:
        parse_unified_diff(diff("@@ -1,0 +1,0 @@\n-a\n+b\n"), limits(max_changed_lines=1))
    assert captured.value.code is PatchErrorCode.TOO_MANY_CHANGED_LINES


@pytest.mark.parametrize("declared", [1, 3, 40])
def test_offset_requires_whole_exact_unique_old_body(declared: int) -> None:
    source = "prefix\nanchor\nold\ntail\nsuffix\n"
    candidate = diff(f"@@ -{declared},3 +{declared},3 @@\n anchor\n-old\n+new\n tail\n")
    assert apply(source, candidate) == b"prefix\nanchor\nnew\ntail\nsuffix\n"


def test_declared_exact_position_wins_even_when_context_repeated_elsewhere() -> None:
    source = "anchor\nold\ntail\ngap\nanchor\nold\ntail\n"
    candidate = diff("@@ -5,3 +5,3 @@\n anchor\n-old\n+new\n tail\n")
    assert apply(source, candidate) == b"anchor\nold\ntail\ngap\nanchor\nnew\ntail\n"


def test_ambiguous_offset_is_rejected() -> None:
    source = "anchor\nold\ntail\ngap\nanchor\nold\ntail\n"
    candidate = diff("@@ -2,3 +2,3 @@\n anchor\n-old\n+new\n tail\n")
    assert_conflict(source, candidate)


@pytest.mark.parametrize(
    "source",
    ["prefix\n anchor\nold\ntail\n", "prefix\nanchor\nwrong\ntail\n", "prefix\nanchor\nold\n"],
)
def test_offset_does_not_fuzz_whitespace_ignore_removed_lines_or_drop_context(source: str) -> None:
    candidate = diff("@@ -8,3 +8,3 @@\n anchor\n-old\n+new\n tail\n")
    assert_conflict(source, candidate)


def test_contextless_replacement_and_pure_insertion_are_not_relocated() -> None:
    assert_conflict("prefix\nold\n", diff("@@ -1 +1 @@\n-old\n+new\n"))
    assert_conflict("prefix\nold\n", diff("@@ -4,0 +5,1 @@\n+new\n"))
    assert apply("prefix\nold\n", diff("@@ -1,0 +2,1 @@\n+new\n")) == b"prefix\nnew\nold\n"


def test_two_offset_hunks_match_original_file_and_preserve_order_after_insertion() -> None:
    source = "prefix\nA\nold1\nB\ngap\nC\nold2\nD\n"
    candidate = diff(
        "@@ -1,3 +1,4 @@\n A\n-old1\n+new1\n+inserted\n B\n",
        "@@ -5,3 +6,3 @@\n C\n-old2\n+new2\n D\n",
    )
    assert apply(source, candidate) == b"prefix\nA\nnew1\ninserted\nB\ngap\nC\nnew2\nD\n"


def test_relocated_hunks_cannot_reverse_actual_source_order() -> None:
    source = "A\nold1\nB\ngap\nC\nold2\nD\n"
    candidate = diff(
        "@@ -1,3 +1,3 @@\n C\n-old2\n+new2\n D\n",
        "@@ -5,3 +5,3 @@\n A\n-old1\n+new1\n B\n",
    )
    assert_conflict(source, candidate)


def test_relocated_hunks_cannot_overlap_actual_source_ranges() -> None:
    source = "A\nold1\nB\nold2\nC\n"
    candidate = diff(
        "@@ -7,3 +7,3 @@\n A\n-old1\n+new1\n B\n",
        "@@ -10,3 +10,3 @@\n B\n-old2\n+new2\n C\n",
    )
    assert_conflict(source, candidate)


def test_offset_search_includes_already_consumed_original_regions_for_ambiguity() -> None:
    source = "A\nold\nB\ngap\nA\nold\nB\n"
    candidate = diff(
        "@@ -1,3 +1,3 @@\n A\n-old\n+new1\n B\n",
        "@@ -9,3 +9,3 @@\n A\n-old\n+new2\n B\n",
    )
    assert_conflict(source, candidate)


def test_later_hunk_cannot_match_text_created_by_an_earlier_hunk() -> None:
    source = "A\nold\nB\ngap\n"
    candidate = diff(
        "@@ -1,3 +1,3 @@\n A\n-old\n+new\n B\n",
        "@@ -9,3 +9,3 @@\n A\n-new\n+newer\n B\n",
    )
    assert_conflict(source, candidate)


def test_sandbox_defense_in_depth_still_requires_canonical_counts() -> None:
    implementation = helper()
    with pytest.raises(implementation["PatchFailure"]) as captured:
        implementation["parse_patch"](VALID_DIFF.replace("-1,2", "-1,99"), 20, 100, 2_000)
    assert captured.value.code == "invalid_diff"


def run_filesystem_helper(tmp_path: Path, candidate: str) -> subprocess.CompletedProcess[str]:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    name = "11111111-1111-4111-8111-111111111111.patch.diff"
    parsed = parse_unified_diff(candidate, limits())
    (artifacts / name).write_bytes(parsed.encoded)
    script = APPLY_PATCH_HELPER.replace('"/workspace/artifacts"', repr(str(artifacts))).replace(
        '"/workspace/repository"', repr(str(tmp_path / "repository"))
    )
    return subprocess.run(
        (
            sys.executable,
            "-I",
            "-c",
            script,
            name,
            str(len(parsed.encoded)),
            parsed.summary.sha256,
            "20",
            "100",
            "2000",
        ),
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_full_helper_applies_normalized_exact_offset_without_changing_permissions(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    target = repository / "module.py"
    target.write_text("prefix\nanchor\nold\ntail\n")
    target.chmod(0o640)
    result = run_filesystem_helper(
        tmp_path, diff("@@ -1,99 +1,99 @@\n anchor\n-old\n+new\n tail\n")
    )
    assert result.returncode == 0, result.stderr
    assert target.read_text() == "prefix\nanchor\nnew\ntail\n"
    assert target.stat().st_mode & 0o777 == 0o640
    assert sorted(item.name for item in repository.iterdir()) == ["module.py"]


def test_full_helper_leaves_all_files_unchanged_if_later_file_conflicts(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    original = "prefix\nanchor\nold\ntail\n"
    (repository / "module.py").write_text(original)
    (repository / "second.py").write_text("not the expected old context\n")
    candidate = diff("@@ -1,99 +1,99 @@\n anchor\n-old\n+new\n tail\n") + diff(
        "@@ -1,3 +1,3 @@\n other\n-before\n+after\n context\n", path="second.py"
    )
    result = run_filesystem_helper(tmp_path, candidate)
    assert result.returncode == 45
    assert result.stderr == "repopilot_patch_error:apply_conflict\n"
    assert (repository / "module.py").read_text() == original
    assert (repository / "second.py").read_text() == "not the expected old context\n"
    assert sorted(item.name for item in repository.iterdir()) == ["module.py", "second.py"]


@pytest.mark.parametrize("parent_symlink", [False, True])
def test_full_helper_never_follows_symlink_even_when_context_matches_exactly(
    tmp_path: Path, parent_symlink: bool
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    original = "prefix\nanchor\nold\ntail\n"
    target = outside / "module.py"
    target.write_text(original)
    if parent_symlink:
        (repository / "linked").symlink_to(outside, target_is_directory=True)
        path = "linked/module.py"
    else:
        (repository / "module.py").symlink_to(target)
        path = "module.py"
    result = run_filesystem_helper(
        tmp_path, diff("@@ -1,3 +1,3 @@\n anchor\n-old\n+new\n tail\n", path=path)
    )
    assert result.returncode == 45
    assert result.stderr == "repopilot_patch_error:apply_conflict\n"
    assert target.read_text() == original
