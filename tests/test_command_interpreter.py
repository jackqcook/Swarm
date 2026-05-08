"""
tests/test_command_interpreter.py
──────────────────────────────────
Unit tests for CommandInterpreter + KnowNoGate.
All tests use the mock VLM backend — no GPU required.
"""

import pytest
import asyncio

from drone.command_interpreter import (
    CommandInterpreter,
    KnowNoGate,
    InterpretResult,
    DroneGoal,
)


# ── KnowNoGate tests ──────────────────────────────────────────────────────────

class TestKnowNoGate:

    def test_singleton_high_confidence(self):
        gate = KnowNoGate(tau=0.6)
        probs = {"A": 0.80, "B": 0.10, "C": 0.05, "D": 0.03, "E": 0.02}
        needs_clarification, pred_set = gate.decide(probs)
        assert not needs_clarification
        assert pred_set == ["A"]

    def test_ambiguous_no_dominant_option(self):
        gate = KnowNoGate(tau=0.6)
        probs = {"A": 0.30, "B": 0.25, "C": 0.25, "D": 0.15, "E": 0.05}
        needs_clarification, pred_set = gate.decide(probs)
        assert needs_clarification

    def test_ask_option_triggers_clarification(self):
        gate = KnowNoGate(tau=0.3)  # Low tau
        probs = {"A": 0.35, "B": 0.20, "C": 0.10, "D": 0.05, "E": 0.30}
        needs_clarification, pred_set = gate.decide(probs)
        assert needs_clarification
        assert "E" in pred_set

    def test_exactly_at_threshold(self):
        gate = KnowNoGate(tau=0.60)
        # Exactly at threshold — should be included
        probs = {"A": 0.60, "B": 0.20, "C": 0.10, "D": 0.08, "E": 0.02}
        needs_clarification, pred_set = gate.decide(probs)
        assert not needs_clarification
        assert "A" in pred_set


# ── CommandInterpreter tests ──────────────────────────────────────────────────

@pytest.fixture
def interpreter():
    return CommandInterpreter(config={
        "model_id": "mock",
        "device": "cpu",
        "knowno_tau": 0.60,
    })


@pytest.mark.asyncio
async def test_return_command_executes(interpreter):
    result = await interpreter.interpret("return to home base")
    assert isinstance(result, InterpretResult)
    assert not result.needs_clarification
    assert result.goal is not None
    assert result.goal.action == "return"


@pytest.mark.asyncio
async def test_hover_command_executes(interpreter):
    result = await interpreter.interpret("hover here for a minute")
    assert not result.needs_clarification
    assert result.goal.action == "hover"


@pytest.mark.asyncio
async def test_ambiguous_command_asks(interpreter):
    result = await interpreter.interpret("unclear go to the thing over there somewhere?")
    assert result.needs_clarification
    assert result.clarification_question is not None
    assert len(result.clarification_question) > 10


@pytest.mark.asyncio
async def test_clarification_includes_options(interpreter):
    result = await interpreter.interpret("unclear go to the thing over there somewhere?")
    if result.needs_clarification:
        # Should include at least one lettered option
        assert any(f"{c})" in result.clarification_question for c in "ABCD")


@pytest.mark.asyncio
async def test_generic_fly_command(interpreter):
    result = await interpreter.interpret("fly northeast and check the field")
    # With mock VLM and default tau=0.6, generic commands have A at 0.55 — ambiguous
    # Just verify it returns a valid result either way
    assert isinstance(result, InterpretResult)
    if not result.needs_clarification:
        assert result.goal is not None
        assert result.goal.action in ("fly_to", "survey_area", "orbit", "hover")


@pytest.mark.asyncio
async def test_result_has_confidence(interpreter):
    result = await interpreter.interpret("return home")
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_result_has_options_list(interpreter):
    result = await interpreter.interpret("hover")
    assert isinstance(result.options_considered, list)
    assert len(result.options_considered) >= 4  # A, B, C, D, E
