from src.hybrid import merge_results


def test_merge_dedupes_by_id_keeping_first():
    vector = [{"id": "claims/a", "text": "claim A", "score": 0.9}]
    graph = [{"id": "claims/a", "text": "claim A", "via": "supports"},
             {"id": "studies/b", "text": "study B", "via": "supports"}]
    merged = merge_results(vector, graph)
    ids = [m["id"] for m in merged]
    assert ids == ["claims/a", "studies/b"]


def test_merge_respects_limit():
    vector = [{"id": f"claims/{i}", "text": "x", "score": 1.0 - i / 10} for i in range(10)]
    merged = merge_results(vector, [], limit=4)
    assert len(merged) == 4
