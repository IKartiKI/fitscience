"""Dynamic ingestion: PDF -> text -> chunks -> entity extraction -> knowledge graph."""
import json
import re

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.db import get_db, ensure_collections
from src.embeddings import embed_texts
from src.prompts import EXTRACTION_TEMPLATE, VERDICT_TEMPLATE

CHUNK_SIZE = 2000  # characters, roughly 500 tokens
CHUNK_OVERLAP = 200

VALID_VERDICTS = {"CONTRADICT", "AGREE", "UNRELATED"}

# Existing claims most similar to a new claim — candidates for contradiction checks.
# Claims from the same paper are excluded so a paper can't contradict itself.
SIMILAR_CLAIMS_AQL = """
FOR c IN claims
  FILTER c.embedding != null AND NOT STARTS_WITH(c._key, @paper_key)
  LET score = COSINE_SIMILARITY(c.embedding, @qvec)
  FILTER score > @min_score
  SORT score DESC
  LIMIT @top_k
  RETURN { id: c._id, text: c.text }
"""


def extract_text(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    with fitz.open(pdf_path) as doc:
        return "\n".join(page.get_text() for page in doc)


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)


def parse_extraction_json(raw: str) -> dict:
    """Parse Gemini's JSON reply, tolerating markdown code fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    return json.loads(cleaned)


def extract_entities(chunk: str, llm) -> dict:
    reply = llm.invoke(EXTRACTION_TEMPLATE.format(chunk_text=chunk)).content
    return parse_extraction_json(reply)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", text.lower().replace(" ", "_"))[:60]


def normalize_verdict(raw) -> str:
    """Coerce the LLM's comparison reply to a valid verdict; default to UNRELATED."""
    verdict = (raw or "").strip().upper()
    return verdict if verdict in VALID_VERDICTS else "UNRELATED"


def link_contradictions(db, llm, study_id: str, paper_key: str, claim_text: str,
                        claim_vec: list[float], top_k: int = 3, min_score: float = 0.55) -> int:
    """Compare a new claim against the most similar existing claims; when the LLM
    judges them contradictory, create a `contradicts` edge from the new study to
    the existing claim. Returns how many contradictions were found."""
    candidates = db.aql.execute(
        SIMILAR_CLAIMS_AQL,
        bind_vars={"paper_key": paper_key, "qvec": claim_vec, "min_score": min_score, "top_k": top_k},
    )
    found = 0
    for cand in candidates:
        reply = llm.invoke(VERDICT_TEMPLATE.format(claim_a=claim_text, claim_b=cand["text"])).content
        if normalize_verdict(reply) == "CONTRADICT":
            existing_key = cand["id"].split("/")[1]
            db.collection("contradicts").insert(
                {"_key": f"{paper_key}__{existing_key}"[:254], "_from": study_id, "_to": cand["id"]},
                overwrite=True,
            )
            found += 1
    return found


def ingest_paper(pdf_path: str, paper_key: str, db=None, llm=None) -> dict:
    """Full pipeline: returns a summary dict of what was inserted."""
    from src.rag import get_llm  # local import to avoid circular dependency

    db = db or get_db()
    llm = llm or get_llm()
    ensure_collections(db)

    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    # Extract entities from the first few chunks (title/claims usually appear early).
    title, year, authors, all_claims = None, None, [], []
    for chunk in chunks[:4]:
        try:
            data = extract_entities(chunk, llm)
        except (json.JSONDecodeError, ValueError):
            continue
        title = title or data.get("title")
        year = year or data.get("year")
        authors = authors or data.get("authors", [])
        all_claims.extend(data.get("claims", []))

    # Insert the study node.
    summary = " ".join(all_claims[:3]) or text[:500]
    study_vec = embed_texts([summary])[0]
    db.collection("studies").insert(
        {"_key": paper_key, "title": title or paper_key, "year": year,
         "authors": authors, "summary": summary, "embedding": study_vec},
        overwrite=True,
    )
    study_id = f"studies/{paper_key}"

    # Insert claim nodes + supports edges; check each new claim against existing
    # science and auto-create contradicts edges where the LLM finds a conflict.
    new_claims, contradictions_found = 0, 0
    if all_claims:
        claim_vecs = embed_texts(all_claims)
        for claim_text, vec in zip(all_claims, claim_vecs):
            claim_key = f"{paper_key}_{_slugify(claim_text)[:40]}"
            contradictions_found += link_contradictions(db, llm, study_id, paper_key, claim_text, vec)
            db.collection("claims").insert(
                {"_key": claim_key, "text": claim_text, "confidence": "extracted", "embedding": vec},
                overwrite=True,
            )
            db.collection("supports").insert(
                {"_key": f"{paper_key}__{claim_key}"[:254], "_from": study_id, "_to": f"claims/{claim_key}"},
                overwrite=True,
            )
            new_claims += 1

    # Insert chunk nodes for vector search + has_chunk edges.
    chunk_vecs = embed_texts(chunks)
    for i, (chunk, vec) in enumerate(zip(chunks, chunk_vecs)):
        chunk_key = f"{paper_key}_chunk_{i}"
        db.collection("chunks").insert(
            {"_key": chunk_key, "text": chunk, "source": study_id, "embedding": vec},
            overwrite=True,
        )
        db.collection("has_chunk").insert(
            {"_key": f"{paper_key}__{chunk_key}"[:254], "_from": study_id, "_to": f"chunks/{chunk_key}"},
            overwrite=True,
        )

    return {"study": study_id, "title": title or paper_key, "claims": new_claims,
            "chunks": len(chunks), "contradictions_found": contradictions_found}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    pdf = sys.argv[1]
    key = _slugify(Path(pdf).stem)
    print(ingest_paper(pdf, key))
