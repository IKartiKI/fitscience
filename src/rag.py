"""Basic RAG: vector similarity search in ArangoDB + Gemini answer generation."""
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from src.db import get_db
from src.embeddings import embed_query
from src.prompts import build_answer_prompt

load_dotenv()

LLM_MODEL = "gemini-2.5-flash"

VECTOR_SEARCH_AQL = """
FOR doc IN @@collection
  FILTER doc.embedding != null
  LET score = COSINE_SIMILARITY(doc.embedding, @qvec)
  SORT score DESC
  LIMIT @top_k
  RETURN { id: doc._id, text: doc.summary || doc.text || doc.name, score: score }
"""


def get_llm():
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


def vector_search(db, query: str, collections=("studies", "claims", "chunks"), top_k=3, qvec=None) -> list[dict]:
    """Search each collection by cosine similarity, return all hits sorted by score."""
    qvec = qvec if qvec is not None else embed_query(query)
    hits = []
    for coll in collections:
        cursor = db.aql.execute(
            VECTOR_SEARCH_AQL,
            bind_vars={"@collection": coll, "qvec": qvec, "top_k": top_k},
        )
        hits.extend(cursor)
    return sorted(hits, key=lambda h: h["score"], reverse=True)


def format_context(hits: list[dict]) -> str:
    """One line per hit; graph hits also show the exact edge that connected them."""
    lines = []
    for h in hits:
        rel = ""
        if h.get("via") and h.get("edge_from"):
            rel = f" (relationship: {h['edge_from']} --{h['via']}--> {h['edge_to']})"
        lines.append(f"[{h['id']}]{rel} {h['text']}")
    return "\n".join(lines)


def answer(query: str, db=None) -> str:
    """Basic RAG: retrieve by vector similarity, then generate."""
    db = db or get_db()
    hits = vector_search(db, query)
    prompt = build_answer_prompt(query=query, context=format_context(hits[:6]), contradictions="")
    return get_llm().invoke(prompt).content


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What does research say about rep ranges for hypertrophy?"
    print(f"Q: {q}\n")
    print(answer(q))
