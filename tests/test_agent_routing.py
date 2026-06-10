from src.agent import normalize_plan


def test_normalize_plan_accepts_valid_values():
    assert normalize_plan("vector") == "vector"
    assert normalize_plan(" Graph \n") == "graph"
    assert normalize_plan("HYBRID") == "hybrid"


def test_normalize_plan_defaults_to_hybrid_on_garbage():
    assert normalize_plan("I think vector search is best") == "hybrid"
    assert normalize_plan("") == "hybrid"
