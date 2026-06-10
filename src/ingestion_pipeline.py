"""Dynamic ingestion: PDF -> text -> chunks -> entity extraction -> knowledge graph."""
import json
import re

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.db import get_db, ensure_collections
from src.embeddings import embed_texts
from src.prompts import EXTRACTION_TEMPLATE

CHUNK_SIZE = 2000  # characters, roughly 500 tokens
CHUNK_OVERLAP = 200


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

    # Insert claim nodes + supports edges.
    new_claims = 0
    if all_claims:
        claim_vecs = embed_texts(all_claims)
        for claim_text, vec in zip(all_claims, claim_vecs):
            claim_key = f"{paper_key}_{_slugify(claim_text)[:40]}"
            db.collection("claims").insert(
                {"_key": claim_key, "text": claim_text, "confidence": "extracted", "embedding": vec},
                overwrite=True,
            )
            db.collection("supports").insert(
                {"_from": study_id, "_to": f"claims/{claim_key}"}, overwrite=True, overwrite_mode="ignore",
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
            {"_from": study_id, "_to": f"chunks/{chunk_key}"}, overwrite=True, overwrite_mode="ignore",
        )

    return {"study": study_id, "title": title, "claims": new_claims, "chunks": len(chunks)}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    pdf = sys.argv[1]
    key = _slugify(Path(pdf).stem)
    print(ingest_paper(pdf, key))
