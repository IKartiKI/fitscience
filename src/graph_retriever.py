"""Graph traversal retrieval: follow typed edges in the knowledge graph."""
from src.db import get_db

# From any start node, walk up to 2 edges in any direction across our edge types,
# returning each related node once with the edge type that connected it.
TRAVERSAL_AQL = """
FOR v, e IN 1..2 ANY @start_id supports, contradicts, cites, applies_to, targets
  RETURN DISTINCT {
    id: v._id,
    text: v.summary || v.text || v.name,
    via: PARSE_IDENTIFIER(e._id).collection
  }
"""

# Find studies that contradict a given claim.
CONTRADICTIONS_AQL = """
FOR study IN 1..1 INBOUND @claim_id contradicts
  RETURN { id: study._id, text: study.summary, title: study.title }
"""

# Resolve a free-text entity mention to a node by vector similarity over one collection.
ENTITY_LOOKUP_AQL = """
FOR doc IN @@collection
  FILTER doc.embedding != null
  LET score = COSINE_SIMILARITY(doc.embedding, @qvec)
  SORT score DESC
  LIMIT 1
  RETURN { id: doc._id, score: score }
"""


def traverse(db, start_id: str) -> list[dict]:
    """Return all nodes within 2 hops of start_id."""
    return list(db.aql.execute(TRAVERSAL_AQL, bind_vars={"start_id": start_id}))


def find_contradictions(db, claim_ids: list[str]) -> list[dict]:
    """Return studies connected to any of these claims via a `contradicts` edge."""
    results = []
    for claim_id in claim_ids:
        if not claim_id.startswith("claims/"):
            continue
        results.extend(db.aql.execute(CONTRADICTIONS_AQL, bind_vars={"claim_id": claim_id}))
    return results


def resolve_entity(db, qvec: list[float], collection: str) -> str | None:
    """Find the best-matching node id in a collection for an embedded query."""
    hits = list(db.aql.execute(ENTITY_LOOKUP_AQL, bind_vars={"@collection": collection, "qvec": qvec}))
    return hits[0]["id"] if hits else None


def graph_search(db, query: str, qvec: list[float]) -> list[dict]:
    """Resolve the query to its closest exercise or claim, then traverse from it."""
    results = []
    for coll in ("exercises", "claims"):
        start = resolve_entity(db, qvec, coll)
        if start:
            results.extend(traverse(db, start))
    seen, unique = set(), []
    for r in results:
        if r["id"] not in seen:
            seen.add(r["id"])
            unique.append(r)
    return unique


if __name__ == "__main__":
    db = get_db()
    print("Traversal from exercises/bench_press:")
    for r in traverse(db, "exercises/bench_press"):
        print(f"  via {r['via']}: {r['id']}")
    print("\nContradictions for claims/volume_dose_response:")
    for r in find_contradictions(db, ["claims/volume_dose_response"]):
        print(f"  {r['id']}: {r['title']}")
