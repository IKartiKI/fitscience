from src.prompts import build_answer_prompt


def test_answer_prompt_includes_context_and_query():
    prompt = build_answer_prompt(
        query="best rep range?",
        context="[claims/rep_range_wide] 6-30 reps work.",
        contradictions="",
    )
    assert "best rep range?" in prompt
    assert "6-30 reps work." in prompt
    assert "conflicting evidence" not in prompt.lower()


def test_answer_prompt_includes_contradictions_when_present():
    prompt = build_answer_prompt(
        query="is volume king?",
        context="[claims/volume_dose_response] more sets, more growth.",
        contradictions="[studies/ralston_2017] found small differences only.",
    )
    assert "ralston_2017" in prompt
    assert "conflicting" in prompt.lower()
