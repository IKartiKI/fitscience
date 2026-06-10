# FitScience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph agent that answers science-based lifting questions using hybrid RAG (vector search + graph traversal) over a fitness knowledge graph in ArangoDB, with dynamic PDF ingestion and contradiction detection.

**Architecture:** ArangoDB stores nodes (studies, claims, exercises, muscle groups, chunks) and typed edges (supports, contradicts, targets, applies_to). A LangGraph agent classifies queries, routes to vector/graph/hybrid retrieval, checks for contradicting studies, and generates cited answers via Gemini. A separate ingestion pipeline turns uploaded PDFs into graph nodes automatically.

**Tech Stack:** Python 3.10+, langchain, langchain-google-genai (Gemini), langgraph, python-arango, pymupdf, streamlit, pytest.

**Spec:** `docs/fitscience-project-design.md`

**Project root:** `d:\Kartik\New folder\ML` (this folder becomes the repo root; no `fitscience/` subfolder).

---

## Prerequisites (manual, do once before Task 0)

1. **Gemini API key:** Go to https://aistudio.google.com/apikey → Create API key. Save it.
2. **ArangoDB Cloud:** Sign up at https://cloud.arangodb.com (free trial tier) → create a deployment → note the **endpoint URL** (looks like `https://xxxx.arangodb.cloud:8529`), **root password**. In the ArangoDB web UI, create a database named `fitscience`.
   - Alternative if cloud signup fails: install Docker and run `docker run -e ARANGO_ROOT_PASSWORD=test123 -p 8529:8529 arangodb/arangodb:3.12` then create the `fitscience` database at http://localhost:8529.

---

### Task 0: Project Scaffold

**Files:**
- Create: `.gitignore`, `requirements.txt`, `.env.example`, `src/__init__.py`, `tests/__init__.py`, `data/papers/.gitkeep`

- [ ] **Step 1: Initialize git repo**

```bash
git init
git add docs/
git commit -m "docs: add FitScience design doc

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Create `.gitignore`**

```gitignore
.env
__pycache__/
*.pyc
.venv/
venv/
.pytest_cache/
data/papers/*.pdf
notebooks/.ipynb_checkpoints/
```

- [ ] **Step 3: Create `requirements.txt`**

```
langchain
langchain-google-genai
langgraph
python-arango
pymupdf
streamlit
python-dotenv
pytest
```

- [ ] **Step 4: Create `.env.example`** (the real `.env` is created by the user, never committed)

```
GOOGLE_API_KEY=your-gemini-api-key-here
ARANGO_URL=https://xxxx.arangodb.cloud:8529
ARANGO_DB=fitscience
ARANGO_USER=root
ARANGO_PASSWORD=your-arango-password-here
```

- [ ] **Step 5: Create the virtual environment and install dependencies**

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Expected: all packages install without errors.

- [ ] **Step 6: Create package markers and folders**

Create empty files: `src/__init__.py`, `tests/__init__.py`, `data/papers/.gitkeep`.

- [ ] **Step 7: Copy `.env.example` to `.env` and fill in real values** (user does this manually with their keys)

- [ ] **Step 8: Commit**

```bash
git add .gitignore requirements.txt .env.example src/__init__.py tests/__init__.py data/papers/.gitkeep
git commit -m "chore: project scaffold

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 1: Seed Knowledge Data

**Files:**
- Create: `data/fitness_knowledge.json`

- [ ] **Step 1: Create `data/fitness_knowledge.json`** with this exact content:

```json
{
  "muscle_groups": [
    {"_key": "chest", "name": "Chest (Pectoralis Major)", "function": "horizontal adduction, shoulder flexion"},
    {"_key": "back", "name": "Back (Lats, Traps, Rhomboids)", "function": "pulling, scapular retraction"},
    {"_key": "quads", "name": "Quadriceps", "function": "knee extension"},
    {"_key": "hamstrings", "name": "Hamstrings", "function": "knee flexion, hip extension"},
    {"_key": "glutes", "name": "Glutes", "function": "hip extension, abduction"},
    {"_key": "shoulders", "name": "Shoulders (Deltoids)", "function": "shoulder abduction, flexion, rotation"},
    {"_key": "biceps", "name": "Biceps", "function": "elbow flexion, supination"},
    {"_key": "triceps", "name": "Triceps", "function": "elbow extension"}
  ],
  "exercises": [
    {"_key": "bench_press", "name": "Barbell Bench Press", "category": "compound", "movement_pattern": "horizontal_push"},
    {"_key": "squat", "name": "Barbell Back Squat", "category": "compound", "movement_pattern": "squat"},
    {"_key": "deadlift", "name": "Conventional Deadlift", "category": "compound", "movement_pattern": "hip_hinge"},
    {"_key": "pull_up", "name": "Pull-Up", "category": "compound", "movement_pattern": "vertical_pull"},
    {"_key": "barbell_row", "name": "Barbell Row", "category": "compound", "movement_pattern": "horizontal_pull"},
    {"_key": "overhead_press", "name": "Overhead Press", "category": "compound", "movement_pattern": "vertical_push"},
    {"_key": "dumbbell_fly", "name": "Dumbbell Fly", "category": "isolation", "movement_pattern": "horizontal_adduction"},
    {"_key": "leg_press", "name": "Leg Press", "category": "compound", "movement_pattern": "squat"},
    {"_key": "romanian_deadlift", "name": "Romanian Deadlift", "category": "compound", "movement_pattern": "hip_hinge"},
    {"_key": "hip_thrust", "name": "Barbell Hip Thrust", "category": "compound", "movement_pattern": "hip_extension"}
  ],
  "claims": [
    {"_key": "volume_dose_response", "text": "There is a dose-response relationship between weekly training volume and hypertrophy: 10-20 hard sets per muscle per week produces more growth than fewer sets.", "confidence": "high"},
    {"_key": "rep_range_wide", "text": "Rep ranges from roughly 6 to 30 reps per set are similarly effective for hypertrophy when sets are taken close to failure.", "confidence": "high"},
    {"_key": "frequency_2x", "text": "Training a muscle group at least twice per week produces greater hypertrophy than training it once per week when volume is equated per session.", "confidence": "moderate"},
    {"_key": "progressive_overload", "text": "Progressive overload (gradually increasing load, reps, or sets over time) is the primary driver of long-term muscle growth.", "confidence": "high"},
    {"_key": "protein_intake", "text": "A daily protein intake of 1.6-2.2 g per kg of bodyweight is sufficient to maximize muscle protein synthesis for most lifters.", "confidence": "high"},
    {"_key": "failure_proximity", "text": "Training close to failure (0-3 reps in reserve) is more important for hypertrophy than the specific load used.", "confidence": "moderate"}
  ],
  "studies": [
    {"_key": "schoenfeld_2017", "title": "Dose-response relationship between weekly resistance training volume and increases in muscle mass: A systematic review and meta-analysis", "authors": ["Brad Schoenfeld", "Dan Ogborn", "James Krieger"], "year": 2017, "journal": "Journal of Sports Sciences", "summary": "Meta-analysis of 15 studies showing higher weekly set volumes produce greater hypertrophy, with 10+ sets per muscle per week outperforming lower volumes in a graded dose-response manner."},
    {"_key": "krieger_2010", "title": "Single vs. multiple sets of resistance exercise for muscle hypertrophy: a meta-analysis", "authors": ["James Krieger"], "year": 2010, "journal": "Journal of Strength and Conditioning Research", "summary": "Meta-analysis finding multiple sets per exercise produce 40% greater hypertrophy than single sets, supporting higher training volumes."},
    {"_key": "helms_2014", "title": "Recommendations for natural bodybuilding contest preparation: resistance and cardiovascular training", "authors": ["Eric Helms", "Peter Fitschen", "Alan Aragon"], "year": 2014, "journal": "Journal of Sports Medicine and Physical Fitness", "summary": "Review concluding that a wide range of rep ranges builds muscle effectively in natural athletes, recommending the majority of training in the 6-12 rep range with some work above and below."},
    {"_key": "schoenfeld_2016", "title": "Effects of resistance training frequency on measures of muscle hypertrophy: A systematic review and meta-analysis", "authors": ["Brad Schoenfeld", "Dan Ogborn", "James Krieger"], "year": 2016, "journal": "Sports Medicine", "summary": "Meta-analysis showing training muscle groups twice per week produces superior hypertrophy compared to once per week."},
    {"_key": "morton_2016", "title": "Neither load nor systemic hormones determine resistance training-mediated hypertrophy or strength gains in resistance-trained young men", "authors": ["Robert Morton", "Sara Oikawa", "Stuart Phillips"], "year": 2016, "journal": "Journal of Applied Physiology", "summary": "RCT showing high-rep light loads and low-rep heavy loads taken to failure produce equivalent hypertrophy, and that post-exercise hormone spikes do not predict growth."},
    {"_key": "ralston_2017", "title": "The effect of weekly set volume on strength gain: a meta-analysis", "authors": ["Grant Ralston", "Lon Kilgore", "Frank Wyatt"], "year": 2017, "journal": "Sports Medicine", "summary": "Meta-analysis finding only small differences between low, medium, and high weekly set volumes for strength outcomes, suggesting diminishing returns from very high volumes."}
  ],
  "edges": {
    "supports": [
      ["studies/schoenfeld_2017", "claims/volume_dose_response"],
      ["studies/krieger_2010", "claims/volume_dose_response"],
      ["studies/helms_2014", "claims/rep_range_wide"],
      ["studies/morton_2016", "claims/rep_range_wide"],
      ["studies/morton_2016", "claims/failure_proximity"],
      ["studies/schoenfeld_2016", "claims/frequency_2x"],
      ["studies/helms_2014", "claims/protein_intake"]
    ],
    "contradicts": [
      ["studies/ralston_2017", "claims/volume_dose_response"]
    ],
    "cites": [
      ["studies/schoenfeld_2017", "studies/krieger_2010"]
    ],
    "applies_to": [
      ["claims/rep_range_wide", "exercises/bench_press"],
      ["claims/rep_range_wide", "exercises/squat"],
      ["claims/rep_range_wide", "exercises/leg_press"],
      ["claims/volume_dose_response", "exercises/bench_press"],
      ["claims/volume_dose_response", "exercises/squat"],
      ["claims/frequency_2x", "exercises/bench_press"],
      ["claims/progressive_overload", "exercises/deadlift"],
      ["claims/progressive_overload", "exercises/overhead_press"]
    ],
    "targets": [
      ["exercises/bench_press", "muscle_groups/chest"],
      ["exercises/bench_press", "muscle_groups/triceps"],
      ["exercises/bench_press", "muscle_groups/shoulders"],
      ["exercises/squat", "muscle_groups/quads"],
      ["exercises/squat", "muscle_groups/glutes"],
      ["exercises/deadlift", "muscle_groups/hamstrings"],
      ["exercises/deadlift", "muscle_groups/glutes"],
      ["exercises/deadlift", "muscle_groups/back"],
      ["exercises/pull_up", "muscle_groups/back"],
      ["exercises/pull_up", "muscle_groups/biceps"],
      ["exercises/barbell_row", "muscle_groups/back"],
      ["exercises/barbell_row", "muscle_groups/biceps"],
      ["exercises/overhead_press", "muscle_groups/shoulders"],
      ["exercises/overhead_press", "muscle_groups/triceps"],
      ["exercises/dumbbell_fly", "muscle_groups/chest"],
      ["exercises/leg_press", "muscle_groups/quads"],
      ["exercises/leg_press", "muscle_groups/glutes"],
      ["exercises/romanian_deadlift", "muscle_groups/hamstrings"],
      ["exercises/romanian_deadlift", "muscle_groups/glutes"],
      ["exercises/hip_thrust", "muscle_groups/glutes"]
    ]
  }
}
```

- [ ] **Step 2: Validate the JSON parses**

Run: `.venv\Scripts\python -c "import json; d=json.load(open('data/fitness_knowledge.json')); print(len(d['studies']), 'studies', len(d['claims']), 'claims')"`
Expected: `6 studies 6 claims`

- [ ] **Step 3: Commit**

```bash
git add data/fitness_knowledge.json
git commit -m "feat: add seed fitness knowledge data

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Database Connection Module

**Files:**
- Create: `src/db.py`
- Test: manual smoke script (needs live ArangoDB — no unit test)

- [ ] **Step 1: Create `src/db.py`**

```python
"""ArangoDB connection and schema setup for the FitScience knowledge graph."""
import os

from arango import ArangoClient
from dotenv import load_dotenv

load_dotenv()

NODE_COLLECTIONS = ["studies", "claims", "exercises", "muscle_groups", "chunks"]
EDGE_COLLECTIONS = ["supports", "contradicts", "cites", "applies_to", "targets", "has_chunk"]


def get_db():
    """Connect to the fitscience database using credentials from .env."""
    client = ArangoClient(hosts=os.environ["ARANGO_URL"])
    return client.db(
        os.environ.get("ARANGO_DB", "fitscience"),
        username=os.environ["ARANGO_USER"],
        password=os.environ["ARANGO_PASSWORD"],
    )


def ensure_collections(db):
    """Create all node and edge collections if they don't exist."""
    for name in NODE_COLLECTIONS:
        if not db.has_collection(name):
            db.create_collection(name)
    for name in EDGE_COLLECTIONS:
        if not db.has_collection(name):
            db.create_collection(name, edge=True)
```

- [ ] **Step 2: Smoke-test the connection**

Run: `.venv\Scripts\python -c "from src.db import get_db, ensure_collections; db = get_db(); ensure_collections(db); print('collections:', sorted(c['name'] for c in db.collections() if not c['name'].startswith('_')))"`

Expected: `collections: ['applies_to', 'chunks', 'cites', 'claims', 'contradicts', 'exercises', 'has_chunk', 'muscle_groups', 'studies', 'supports', 'targets']`

If this fails with a connection error: re-check `ARANGO_URL` and password in `.env`, and that the `fitscience` database exists in the ArangoDB web UI.

- [ ] **Step 3: Commit**

```bash
git add src/db.py
git commit -m "feat: ArangoDB connection and collection setup

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Embeddings Module

**Files:**
- Create: `src/embeddings.py`

- [ ] **Step 1: Create `src/embeddings.py`**

```python
"""Gemini embedding helpers. One place to change the model later."""
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

EMBED_MODEL = "models/text-embedding-004"

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of documents."""
    return _get_embedder().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """Embed a single search query."""
    return _get_embedder().embed_query(text)
```

- [ ] **Step 2: Smoke-test embeddings (hits the Gemini API)**

Run: `.venv\Scripts\python -c "from src.embeddings import embed_query; v = embed_query('hypertrophy rep ranges'); print(type(v), len(v))"`
Expected: `<class 'list'> 768`

If this fails with an auth error: check `GOOGLE_API_KEY` in `.env`.

- [ ] **Step 3: Commit**

```bash
git add src/embeddings.py
git commit -m "feat: Gemini embedding helpers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Seed Data Ingestion

**Files:**
- Create: `src/ingest.py`

- [ ] **Step 1: Create `src/ingest.py`**

```python
"""Load seed data from data/fitness_knowledge.json into ArangoDB with embeddings."""
import json
from pathlib import Path

from src.db import get_db, ensure_collections
from src.embeddings import embed_texts

DATA_FILE = Path(__file__).parent.parent / "data" / "fitness_knowledge.json"

# Which field of each node type is the text we embed for vector search.
EMBED_FIELDS = {"studies": "summary", "claims": "text", "exercises": "name"}


def load_seed_data(db, data: dict):
    """Insert all nodes (with embeddings) and edges. Idempotent via overwrite."""
    for coll_name in ["muscle_groups", "exercises", "claims", "studies"]:
        docs = data[coll_name]
        embed_field = EMBED_FIELDS.get(coll_name)
        if embed_field:
            vectors = embed_texts([d[embed_field] for d in docs])
            for doc, vec in zip(docs, vectors):
                doc["embedding"] = vec
        db.collection(coll_name).import_bulk(docs, on_duplicate="replace")
        print(f"  inserted {len(docs)} into {coll_name}")

    for edge_coll, pairs in data["edges"].items():
        edges = [{"_from": f, "_to": t} for f, t in pairs]
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
```

- [ ] **Step 2: Run the ingestion**

Run: `.venv\Scripts\python -m src.ingest`
Expected output ends with:
```
  studies: 6
  claims: 6
  exercises: 10
  muscle_groups: 8
  supports: 7
  contradicts: 1
  targets: 20
```

- [ ] **Step 3: Verify in ArangoDB web UI** — open the `studies` collection, confirm `schoenfeld_2017` exists and has an `embedding` array of 768 floats.

- [ ] **Step 4: Commit**

```bash
git add src/ingest.py
git commit -m "feat: seed data ingestion with embeddings

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Basic RAG Pipeline (vector search + generation)

**Files:**
- Create: `src/prompts.py`, `src/rag.py`
- Test: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test** — `tests/test_prompts.py`:

```python
from src.prompts import build_answer_prompt


def test_answer_prompt_includes_context_and_query():
    prompt = build_answer_prompt(
        query="best rep range?",
        context="[claims/rep_range_wide] 6-30 reps work.",
        contradictions="",
    )
    assert "best rep range?" in prompt
    assert "6-30 reps work." in prompt
    assert "conflicting evidence" not in prompt.lower()


def test_answer_prompt_includes_contradictions_when_present():
    prompt = build_answer_prompt(
        query="is volume king?",
        context="[claims/volume_dose_response] more sets, more growth.",
        contradictions="[studies/ralston_2017] found small differences only.",
    )
    assert "ralston_2017" in prompt
    assert "conflicting" in prompt.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.prompts'`

- [ ] **Step 3: Create `src/prompts.py`**

```python
"""All prompt templates in one place."""

ANSWER_TEMPLATE = """You are FitScience, an evidence-based lifting assistant.
Answer the user's question using ONLY the retrieved context below.
Cite sources inline using their keys, e.g. (schoenfeld_2017).
If the context does not contain the answer, say so honestly.

Retrieved context:
{context}
{contradiction_block}
User question: {query}

Answer:"""

CONTRADICTION_BLOCK = """
Conflicting evidence found in the knowledge graph:
{contradictions}
You MUST mention this conflicting evidence in a short "Note on conflicting evidence" section at the end of your answer.
"""

CLASSIFY_TEMPLATE = """Classify this fitness question and pick a retrieval strategy.

Question: {query}

Strategies:
- "vector": general/semantic question, no specific entity (e.g. "how do muscles grow?")
- "graph": asks about a specific exercise or muscle group and its relationships (e.g. "what exercises target chest?")
- "hybrid": asks for evidence-based advice combining concepts (e.g. "best rep range for squat hypertrophy?")

Reply with EXACTLY one word: vector, graph, or hybrid."""

EXTRACTION_TEMPLATE = """You are a fitness science knowledge extractor.
From the research paper text below, extract structured information.

Text:
{chunk_text}

Return ONLY a valid JSON object (no markdown fences, no commentary) with these keys:
- "title": paper title if visible in this text, else null
- "year": publication year as integer if visible, else null
- "authors": list of author names if visible, else []
- "claims": list of specific scientific claims this text makes (each a single sentence string)
- "exercises": list of exercise names mentioned, else []
- "muscle_groups": list of muscle groups mentioned, else []
"""


def build_answer_prompt(query: str, context: str, contradictions: str) -> str:
    block = CONTRADICTION_BLOCK.format(contradictions=contradictions) if contradictions.strip() else ""
    return ANSWER_TEMPLATE.format(context=context, contradiction_block=block, query=query)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_prompts.py -v`
Expected: 2 passed

- [ ] **Step 5: Create `src/rag.py`**

```python
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


def vector_search(db, query: str, collections=("studies", "claims", "chunks"), top_k=3) -> list[dict]:
    """Search each collection by cosine similarity, return all hits sorted by score."""
    qvec = embed_query(query)
    hits = []
    for coll in collections:
        cursor = db.aql.execute(
            VECTOR_SEARCH_AQL,
            bind_vars={"@collection": coll, "qvec": qvec, "top_k": top_k},
        )
        hits.extend(cursor)
    return sorted(hits, key=lambda h: h["score"], reverse=True)


def format_context(hits: list[dict]) -> str:
    return "\n".join(f"[{h['id']}] {h['text']}" for h in hits)


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
```

- [ ] **Step 6: Smoke-test basic RAG end to end (hits ArangoDB + Gemini)**

Run: `.venv\Scripts\python -m src.rag "What rep range should I use to build muscle?"`
Expected: A coherent answer citing `helms_2014` and/or `morton_2016`. If the LLM call fails with a model-not-found error, list available models with `.venv\Scripts\python -c "import google.generativeai as genai, os; genai.configure(api_key=os.environ['GOOGLE_API_KEY']); [print(m.name) for m in genai.list_models()]"` and update `LLM_MODEL`.

- [ ] **Step 7: Commit**

```bash
git add src/prompts.py src/rag.py tests/test_prompts.py
git commit -m "feat: basic RAG pipeline with vector search and Gemini

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Milestone: Day 1 complete — you now have working basic RAG.**

---

### Task 6: Graph Retriever (traversal + contradiction lookup)

**Files:**
- Create: `src/graph_retriever.py`

- [ ] **Step 1: Create `src/graph_retriever.py`**

```python
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
```

- [ ] **Step 2: Run the smoke test**

Run: `.venv\Scripts\python -m src.graph_retriever`
Expected: traversal lists `muscle_groups/chest`, `muscle_groups/triceps`, `muscle_groups/shoulders`, plus claims reached via `applies_to`; contradictions section lists `studies/ralston_2017`.

- [ ] **Step 3: Commit**

```bash
git add src/graph_retriever.py
git commit -m "feat: graph traversal and contradiction lookup

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Hybrid Retrieval Merge

**Files:**
- Create: `src/hybrid.py`
- Test: `tests/test_hybrid.py`

- [ ] **Step 1: Write the failing test** — `tests/test_hybrid.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_hybrid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.hybrid'`

- [ ] **Step 3: Create `src/hybrid.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_hybrid.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/hybrid.py tests/test_hybrid.py
git commit -m "feat: hybrid retrieval merge with dedupe

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Dynamic PDF Ingestion Pipeline

**Files:**
- Create: `src/ingestion_pipeline.py`
- Test: `tests/test_ingestion.py`

- [ ] **Step 1: Write the failing tests** — `tests/test_ingestion.py`:

```python
from src.ingestion_pipeline import chunk_text, parse_extraction_json


def test_chunk_text_splits_long_text():
    text = "Muscle hypertrophy. " * 500  # ~10,000 chars
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= 2000 for c in chunks)


def test_chunk_text_short_text_is_one_chunk():
    chunks = chunk_text("Short text about squats.")
    assert chunks == ["Short text about squats."]


def test_parse_extraction_json_strips_markdown_fences():
    raw = '```json\n{"title": "T", "year": 2020, "authors": [], "claims": ["c1"], "exercises": [], "muscle_groups": []}\n```'
    data = parse_extraction_json(raw)
    assert data["title"] == "T"
    assert data["claims"] == ["c1"]


def test_parse_extraction_json_handles_plain_json():
    raw = '{"title": null, "year": null, "authors": [], "claims": [], "exercises": [], "muscle_groups": []}'
    assert parse_extraction_json(raw)["title"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_ingestion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingestion_pipeline'`

- [ ] **Step 3: Create `src/ingestion_pipeline.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_ingestion.py -v`
Expected: 4 passed

- [ ] **Step 5: End-to-end pipeline smoke test.** Create a test PDF and ingest it:

```powershell
.venv\Scripts\python -c "import fitz; d = fitz.open(); p = d.new_page(); p.insert_text((72, 72), 'Effects of Stretching on Hypertrophy. Smith et al. 2023. This randomized trial found that loaded inter-set stretching produced an additional 5 percent muscle growth in the calves compared to traditional sets. Stretching under load may enhance hypertrophy.', fontsize=11); d.save('data/papers/test_paper.pdf')"
.venv\Scripts\python -m src.ingestion_pipeline data/papers/test_paper.pdf
```

Expected: prints a dict like `{'study': 'studies/test_paper', 'title': 'Effects of Stretching on Hypertrophy', 'claims': 1+, 'chunks': 1}`.

- [ ] **Step 6: Verify the new paper is searchable**

Run: `.venv\Scripts\python -m src.rag "Does stretching between sets help muscle growth?"`
Expected: answer references the stretching study just ingested.

- [ ] **Step 7: Commit**

```bash
git add src/ingestion_pipeline.py tests/test_ingestion.py
git commit -m "feat: dynamic PDF ingestion pipeline

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Milestone: Day 2 complete — graph retrieval, hybrid merge, and dynamic ingestion all work.**

---

### Task 9: LangGraph Agent

**Files:**
- Create: `src/agent.py`
- Test: `tests/test_agent_routing.py`

- [ ] **Step 1: Write the failing test** — `tests/test_agent_routing.py`:

```python
from src.agent import normalize_plan


def test_normalize_plan_accepts_valid_values():
    assert normalize_plan("vector") == "vector"
    assert normalize_plan(" Graph \n") == "graph"
    assert normalize_plan("HYBRID") == "hybrid"


def test_normalize_plan_defaults_to_hybrid_on_garbage():
    assert normalize_plan("I think vector search is best") == "hybrid"
    assert normalize_plan("") == "hybrid"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_agent_routing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.agent'`

- [ ] **Step 3: Create `src/agent.py`**

```python
"""LangGraph agent: classify -> retrieve (vector/graph/hybrid) -> contradiction check -> answer."""
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.db import get_db
from src.embeddings import embed_query
from src.graph_retriever import find_contradictions, graph_search
from src.hybrid import merge_results
from src.prompts import CLASSIFY_TEMPLATE, build_answer_prompt
from src.rag import format_context, get_llm, vector_search

VALID_PLANS = {"vector", "graph", "hybrid"}


class AgentState(TypedDict):
    query: str
    retrieval_plan: str
    vector_results: list
    graph_results: list
    contradictions: list
    final_answer: str


def normalize_plan(raw: str) -> str:
    """Coerce the LLM's classification reply to a valid plan; default to hybrid."""
    plan = (raw or "").strip().lower()
    return plan if plan in VALID_PLANS else "hybrid"


def make_agent(db=None, llm=None):
    db = db or get_db()
    llm = llm or get_llm()

    def analyze_query(state: AgentState) -> dict:
        reply = llm.invoke(CLASSIFY_TEMPLATE.format(query=state["query"])).content
        return {"retrieval_plan": normalize_plan(reply)}

    def vector_node(state: AgentState) -> dict:
        return {"vector_results": vector_search(db, state["query"]), "graph_results": []}

    def graph_node(state: AgentState) -> dict:
        qvec = embed_query(state["query"])
        return {"graph_results": graph_search(db, state["query"], qvec), "vector_results": []}

    def hybrid_node(state: AgentState) -> dict:
        qvec = embed_query(state["query"])
        return {
            "vector_results": vector_search(db, state["query"]),
            "graph_results": graph_search(db, state["query"], qvec),
        }

    def contradiction_check(state: AgentState) -> dict:
        merged = merge_results(state["vector_results"], state["graph_results"])
        claim_ids = [h["id"] for h in merged if h["id"].startswith("claims/")]
        return {"contradictions": find_contradictions(db, claim_ids)}

    def generate_answer(state: AgentState) -> dict:
        merged = merge_results(state["vector_results"], state["graph_results"])
        contradiction_text = "\n".join(f"[{c['id']}] {c['text']}" for c in state["contradictions"])
        prompt = build_answer_prompt(
            query=state["query"],
            context=format_context(merged),
            contradictions=contradiction_text,
        )
        return {"final_answer": llm.invoke(prompt).content}

    graph = StateGraph(AgentState)
    graph.add_node("analyze_query", analyze_query)
    graph.add_node("vector_search", vector_node)
    graph.add_node("graph_traversal", graph_node)
    graph.add_node("hybrid_search", hybrid_node)
    graph.add_node("contradiction_check", contradiction_check)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "analyze_query")
    graph.add_conditional_edges(
        "analyze_query",
        lambda state: state["retrieval_plan"],
        {"vector": "vector_search", "graph": "graph_traversal", "hybrid": "hybrid_search"},
    )
    for node in ("vector_search", "graph_traversal", "hybrid_search"):
        graph.add_edge(node, "contradiction_check")
    graph.add_edge("contradiction_check", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=MemorySaver())


def ask(agent, query: str, thread_id: str = "default") -> AgentState:
    config = {"configurable": {"thread_id": thread_id}}
    return agent.invoke({"query": query}, config=config)


if __name__ == "__main__":
    import sys
    agent = make_agent()
    q = " ".join(sys.argv[1:]) or "Is more volume always better for muscle growth?"
    result = ask(agent, q)
    print(f"Plan: {result['retrieval_plan']}")
    print(f"Contradictions found: {len(result['contradictions'])}")
    print(f"\n{result['final_answer']}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_agent_routing.py -v`
Expected: 2 passed

- [ ] **Step 5: Smoke-test the full agent (hits ArangoDB + Gemini)**

Run: `.venv\Scripts\python -m src.agent "Is more volume always better for muscle growth?"`
Expected: `Plan: hybrid` (or vector), `Contradictions found: 1` (ralston_2017), and an answer that includes a note on conflicting evidence.

Also run: `.venv\Scripts\python -m src.agent "What exercises target the chest?"`
Expected: `Plan: graph`, answer mentions bench press and dumbbell fly.

- [ ] **Step 6: Run the full test suite**

Run: `.venv\Scripts\python -m pytest -v`
Expected: all tests pass (prompts, hybrid, ingestion, agent routing).

- [ ] **Step 7: Commit**

```bash
git add src/agent.py tests/test_agent_routing.py
git commit -m "feat: LangGraph agent with routing, contradiction check, and memory

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Streamlit UI

**Files:**
- Create: `app.py`

- [ ] **Step 1: Create `app.py`**

```python
"""FitScience demo UI: chat with the agent, upload new research PDFs."""
import tempfile
from pathlib import Path

import streamlit as st

from src.agent import ask, make_agent
from src.ingestion_pipeline import _slugify, ingest_paper

st.set_page_config(page_title="FitScience", page_icon="💪", layout="wide")
st.title("💪 FitScience — Evidence-Based Lifting Assistant")
st.caption("Hybrid GraphRAG over a fitness knowledge graph in ArangoDB")


@st.cache_resource
def get_agent():
    return make_agent()


if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: PDF ingestion
with st.sidebar:
    st.header("📄 Add a research paper")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded and st.button("Ingest into knowledge graph"):
        with st.spinner("Extracting, chunking, embedding..."):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name
            result = ingest_paper(tmp_path, _slugify(Path(uploaded.name).stem))
            Path(tmp_path).unlink(missing_ok=True)
        st.success(f"Ingested **{result['title']}** — {result['claims']} claims, {result['chunks']} chunks")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if query := st.chat_input("Ask a lifting question backed by science..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge graph..."):
            result = ask(get_agent(), query, thread_id="streamlit-session")
        meta = f"`plan: {result['retrieval_plan']}` · `contradictions: {len(result['contradictions'])}`"
        reply = f"{result['final_answer']}\n\n---\n{meta}"
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
```

- [ ] **Step 2: Run the app**

Run: `.venv\Scripts\python -m streamlit run app.py`
Expected: browser opens at http://localhost:8501. Ask "Is more volume always better for hypertrophy?" — answer appears with plan and contradiction count in the footer.

- [ ] **Step 3: Test PDF upload in the UI** — upload `data/papers/test_paper.pdf` from Task 8, click "Ingest into knowledge graph", then ask "Does stretching between sets help growth?" Expected: answer uses the ingested paper.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: Streamlit chat UI with PDF upload

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: README and Demo Prep

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create `README.md`**

```markdown
# 💪 FitScience — Evidence-Based Lifting Assistant

A LangGraph agent that answers gym questions using **hybrid GraphRAG**: vector
search + graph traversal over a fitness research knowledge graph stored in
**ArangoDB**, with answers generated by **Gemini**.

## What makes it interesting

- **Knowledge graph, not flat docs:** studies *support* claims, claims *apply to*
  exercises, exercises *target* muscle groups. The agent traverses these typed
  edges to build richer context than vector search alone.
- **Contradiction detection:** the agent checks `contradicts` edges and flags
  conflicting studies instead of hiding them.
- **Growing knowledge base:** drop in a research PDF and the ingestion pipeline
  chunks it, extracts claims via Gemini, embeds everything, and updates the graph.

## Architecture

User query → LangGraph agent → classify → vector / graph / hybrid retrieval
(ArangoDB) → contradiction check → Gemini → cited answer.

## Setup

1. `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt`
2. Copy `.env.example` to `.env`, fill in Gemini API key + ArangoDB credentials
3. `python -m src.ingest` — load the seed knowledge graph
4. `streamlit run app.py`

## Demo questions

1. "What exercises target the chest?" — *graph traversal*
2. "What rep range builds the most muscle?" — *vector search with citations*
3. "Is more volume always better for muscle growth?" — *contradiction detection fires (ralston_2017)*
4. Upload a new paper PDF, then ask about its topic — *dynamic ingestion*

## Tests

`python -m pytest -v`
```

- [ ] **Step 2: Run the full test suite one final time**

Run: `.venv\Scripts\python -m pytest -v`
Expected: all tests pass.

- [ ] **Step 3: Walk through all 4 demo questions in the Streamlit app** and confirm each behaves as described in the README.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: README with setup and demo guide

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

**Milestone: Day 3 complete — fully working interview demo.**

---

## Notes for the executor

- Tasks 2, 4, 5 (step 6), 6, 8 (steps 5-6), 9 (step 5), 10 require live ArangoDB + Gemini credentials in `.env`. If `.env` is missing, stop and ask the user to complete the Prerequisites section.
- `COSINE_SIMILARITY` is a built-in AQL function (ArangoDB ≥ 3.9). No vector index needed at this data size — brute-force scan is fine for <1000 docs.
- If `gemini-2.5-flash` is unavailable, swap `LLM_MODEL` in `src/rag.py` for any available `gemini-*` chat model (see Task 5 step 6 for how to list models).
- The conversation "memory" is LangGraph's `MemorySaver` checkpointer keyed by `thread_id` — state persists across `ask()` calls within one process. That's sufficient for the demo; explain it honestly in the interview.
```
