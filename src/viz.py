"""Build a visual evidence subgraph from an agent result.

collect_graph_elements() is pure (testable without Streamlit); render_evidence_graph()
turns its output into streamlit-agraph components inside the app.
"""

COLLECTION_COLORS = {
    "studies": "#4C9AFF",       # blue
    "claims": "#FFC400",        # amber
    "exercises": "#36B37E",     # green
    "muscle_groups": "#FF5630", # red-orange
    "chunks": "#998DD9",        # purple
}

CONTRADICT_COLOR = "#FF0000"


def collect_graph_elements(result: dict) -> tuple[list[dict], list[dict]]:
    """Extract unique nodes and edges from an agent result.

    Nodes: {id, collection, text}. Edges: {source, target, label}.
    Graph hits carry real edge endpoints; vector hits appear as standalone nodes;
    contradictions draw a study -> claim edge.
    """
    nodes, edges, seen_edges = {}, [], set()

    def add_node(node_id: str, text: str = ""):
        if node_id and node_id not in nodes:
            nodes[node_id] = {"id": node_id, "collection": node_id.split("/")[0], "text": text}

    def add_edge(source: str, target: str, label: str):
        key = (source, label, target)
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append({"source": source, "target": target, "label": label})

    for hit in result.get("graph_results", []):
        add_node(hit["id"], hit.get("text", ""))
        if hit.get("edge_from") and hit.get("edge_to"):
            add_node(hit["edge_from"])
            add_node(hit["edge_to"])
            add_edge(hit["edge_from"], hit["edge_to"], hit.get("via", ""))

    for hit in result.get("vector_results", [])[:6]:
        add_node(hit["id"], hit.get("text", ""))

    for c in result.get("contradictions", []):
        add_node(c["id"], c.get("text", ""))
        if c.get("claim_id"):
            add_node(c["claim_id"])
            add_edge(c["id"], c["claim_id"], "contradicts")

    return list(nodes.values()), edges


def render_evidence_graph(result: dict):
    """Render the evidence subgraph in Streamlit. No-op if there is nothing to draw."""
    from streamlit_agraph import Config, Edge, Node, agraph

    raw_nodes, raw_edges = collect_graph_elements(result)
    if not raw_nodes:
        return

    nodes = [
        Node(
            id=n["id"],
            label=n["id"].split("/")[1],
            title=n["text"],  # hover tooltip
            color=COLLECTION_COLORS.get(n["collection"], "#cccccc"),
            size=18,
        )
        for n in raw_nodes
    ]
    edges = [
        Edge(
            source=e["source"],
            target=e["target"],
            label=e["label"],
            color=CONTRADICT_COLOR if e["label"] == "contradicts" else "#888888",
        )
        for e in raw_edges
    ]
    config = Config(width=750, height=420, directed=True, physics=True, hierarchical=False)
    agraph(nodes=nodes, edges=edges, config=config)
