from src.viz import collect_graph_elements


def _result(**overrides):
    base = {"vector_results": [], "graph_results": [], "contradictions": []}
    base.update(overrides)
    return base


def test_collects_triples_from_graph_results():
    result = _result(graph_results=[
        {"id": "muscle_groups/chest", "text": "Chest", "via": "targets",
         "edge_from": "exercises/bench_press", "edge_to": "muscle_groups/chest"},
    ])
    nodes, edges = collect_graph_elements(result)
    node_ids = {n["id"] for n in nodes}
    assert {"exercises/bench_press", "muscle_groups/chest"} <= node_ids
    assert edges == [{"source": "exercises/bench_press", "target": "muscle_groups/chest", "label": "targets"}]


def test_dedupes_repeated_edges():
    hit = {"id": "muscle_groups/chest", "text": "Chest", "via": "targets",
           "edge_from": "exercises/bench_press", "edge_to": "muscle_groups/chest"}
    nodes, edges = collect_graph_elements(_result(graph_results=[hit, dict(hit)]))
    assert len(edges) == 1


def test_contradictions_become_red_edges():
    result = _result(contradictions=[
        {"id": "studies/ralston_2017", "text": "small differences", "claim_id": "claims/volume_dose_response"},
    ])
    nodes, edges = collect_graph_elements(result)
    assert edges == [{"source": "studies/ralston_2017", "target": "claims/volume_dose_response",
                      "label": "contradicts"}]


def test_vector_hits_become_standalone_nodes():
    result = _result(vector_results=[{"id": "claims/rep_range_wide", "text": "6-30 reps", "score": 0.9}])
    nodes, edges = collect_graph_elements(result)
    assert nodes[0]["id"] == "claims/rep_range_wide"
    assert nodes[0]["collection"] == "claims"
    assert edges == []
