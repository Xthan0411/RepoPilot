"""A historical test pass must never certify a later candidate revision."""

import sys
from pathlib import Path
from uuid import uuid4

from repopilot.repair_workflow import _fresh_validation_test

sys.path.insert(0, str(Path(__file__).parents[2]))
from tests.unit.test_repair_agent import verified_source


def test_last_validation_is_bound_to_preceding_patch() -> None:
    state, _, _, _ = verified_source()
    assert _fresh_validation_test(state) == state.test_state.last_result


def test_new_patch_invalidates_old_passing_test_even_with_same_diff() -> None:
    state, _, _, _ = verified_source()
    patch = next(result for result in state.tool_results if result.tool_name == "apply_patch")
    assert patch.payload is not None
    updated = state.patch_state.model_copy(update={"revision": state.patch_state.revision + 1})
    payload = {**patch.payload, "patch_state": updated.model_dump(mode="json")}
    second = patch.model_copy(update={"call_id": uuid4(), "payload": payload})
    stale = state.model_copy(
        update={
            "patch_state": updated,
            "tool_results": (*state.tool_results, second),
        }
    )
    assert _fresh_validation_test(stale) is None


def test_validation_before_patch_or_without_patch_is_not_evidence() -> None:
    state, _, _, _ = verified_source()
    patch = next(result for result in state.tool_results if result.tool_name == "apply_patch")
    tests = tuple(result for result in state.tool_results if result.tool_name == "run_tests")
    assert _fresh_validation_test(state.model_copy(update={"tool_results": tests})) is None
    assert (
        _fresh_validation_test(state.model_copy(update={"tool_results": (*tests, patch)})) is None
    )


def test_test_state_call_identity_must_match_history() -> None:
    state, _, _, _ = verified_source()
    assert state.test_state.last_result is not None
    altered = state.test_state.last_result.model_copy(update={"observation_call_id": uuid4()})
    mismatch = state.model_copy(
        update={
            "test_state": state.test_state.model_copy(update={"last_result": altered}),
        }
    )
    assert _fresh_validation_test(mismatch) is None
