"""Load seed data from data/fitness_knowledge.json into ArangoDB with embeddings."""
import json
from pathlib import Path

from src.db import get_db, ensure_collections
from src.embeddings import embed_texts

DATA_FILE = Path(__file__).parent.parent / "data" / "fitness_knowledge.json"

# Which field of each node type is the text we embed for vector search.
EMBED_FIELDS = {"studies": "summary", "claims": "text", "exercises": "name", "muscle_groups": "name"}


def load_seed_data(db, data: dict):
    """Insert all nodes (with embeddings) and edges. Idempotent via overwrite."""
    for coll_name in ["muscle_groups", "exercises", "claims", "studies"]:
        docs = [dict(d) for d in data[coll_name]]
        embed_field = EMBED_FIELDS.get(coll_name)
        if embed_field:
            vectors = embed_texts([d[embed_field] for d in docs])
            for doc, vec in zip(docs, vectors):
                doc["embedding"] = vec
        db.collection(coll_name).import_bulk(docs, on_duplicate="replace")
        print(f"  inserted {len(docs)} into {coll_name}")

    for edge_coll, pairs in data["edges"].items():
        # Deterministic _key makes re-running ingest idempotent (no duplicate edges).
        edges = [
            {"_key": f"{f.split('/')[1]}__{t.split('/')[1]}", "_from": f, "_to": t}
            for f, t in pairs
        ]
        db.collection(edge_coll).import_bulk(edges, on_duplicate="replace")
        print(f"  inserted {len(edges)} edges into {edge_coll}")


def main():
    db = get_db()
    ensure_collections(db)
    data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    load_seed_data(db, data)
    print("Done. Counts:")
    for c in ["studies", "claims", "exercises", "muscle_groups", "supports", "contradicts", "targets"]:
        print(f"  {c}: {db.collection(c).count()}")


if __name__ == "__main__":
    main()
