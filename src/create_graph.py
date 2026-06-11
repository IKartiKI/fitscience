"""Register a named graph so the knowledge graph can be explored visually
in the ArangoDB web UI (GRAPHS section). Purely cosmetic — AQL traversals
in this project work without it."""
from src.db import get_db

GRAPH_NAME = "fitscience_graph"

# Each edge collection with the vertex collections it connects.
EDGE_DEFINITIONS = [
    {"edge_collection": "supports", "from_vertex_collections": ["studies"], "to_vertex_collections": ["claims"]},
    {"edge_collection": "contradicts", "from_vertex_collections": ["studies"], "to_vertex_collections": ["claims"]},
    {"edge_collection": "cites", "from_vertex_collections": ["studies"], "to_vertex_collections": ["studies"]},
    {"edge_collection": "applies_to", "from_vertex_collections": ["claims"], "to_vertex_collections": ["exercises"]},
    {"edge_collection": "targets", "from_vertex_collections": ["exercises"], "to_vertex_collections": ["muscle_groups"]},
    {"edge_collection": "has_chunk", "from_vertex_collections": ["studies"], "to_vertex_collections": ["chunks"]},
]


def main():
    db = get_db()
    if db.has_graph(GRAPH_NAME):
        print(f"Graph '{GRAPH_NAME}' already exists — nothing to do.")
        return
    db.create_graph(GRAPH_NAME, edge_definitions=EDGE_DEFINITIONS)
    print(f"Created named graph '{GRAPH_NAME}'.")
    print("Open the ArangoDB web UI -> GRAPHS to explore it visually.")


if __name__ == "__main__":
    main()
