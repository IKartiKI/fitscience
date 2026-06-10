"""Merge vector-search hits and graph-traversal hits into one context list."""


def merge_results(vector_hits: list[dict], graph_hits: list[dict], limit: int = 10) -> list[dict]:
    """Vector hits first (already sorted by score), then graph hits; dedupe by id."""
    seen, merged = set(), []
    for hit in list(vector_hits) + list(graph_hits):
        if hit["id"] in seen:
            continue
        seen.add(hit["id"])
        merged.append(hit)
        if len(merged) >= limit:
            break
    return merged
