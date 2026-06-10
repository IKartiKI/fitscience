# FitScience — AI Research Agent for Science-Based Lifting
### Project Design & PRD
**Date:** 2026-06-10
**Target:** AI Engineer Interview at ArangoDB (arango.ai)
**Timeline:** 2-3 days | Stack: Python, LangChain, LangGraph, ArangoDB, Gemini API

---

## 1. What You Are Building

A conversational AI agent that answers gym and lifting questions by reasoning over a **knowledge graph** of fitness research stored in ArangoDB. It combines **semantic vector search** (find relevant content by meaning) with **graph traversal** (follow relationships between entities) to give science-backed, cited answers.

**Example interaction:**
> User: "What does research say about optimal rep ranges for hypertrophy?"
> Agent: Searches graph → finds studies on hypertrophy → traverses Study→Claim→Exercise relationships → synthesizes answer with citations from actual research summaries.

**Why this impresses ArangoDB:**
You are not just using a generic vector database. You are using ArangoDB's multi-model capability (graph + vector in one DB), which is exactly what they sell. You will be able to say "I understood your product and built with it."

**What makes this intermediate-level (not a toy):**
- A dynamic ingestion pipeline that can process new PDF/text research papers automatically — the knowledge base grows without manual work
- A contradiction detection agent node that finds conflicting studies and surfaces disagreements — not just retrieval but reasoning
- Conversation memory so the agent maintains context across multiple turns in a session

---

## 2. Concepts You Will Learn (Mapped to Your 5 Goals)

| # | Concept | Where It Appears in This Project |
|---|---------|----------------------------------|
| 1 | What is an Agent | The LangGraph agent that decides whether to search, traverse, contradict-check, or combine — it reasons and takes actions |
| 2 | Agent with LangChain/LangGraph | The entire orchestration layer is LangGraph — you will build nodes, edges, and conditional routing |
| 3 | What is RAG | The vector search step — embed the query, find similar document chunks, feed to Gemini |
| 4 | Basic RAG | Standalone ingestion + retrieval pipeline you build first before adding the agent |
| 5 | Hybrid RAG + GraphRAG | Combining vector search with ArangoDB graph traversal — the final, interview-worthy upgrade |
| 6 | Dynamic Ingestion Pipeline | Automatic PDF/text → chunking → entity extraction → graph insertion — knowledge base grows without manual work |
| 7 | Contradiction Detection | Agent node that finds studies linked by `contradicts` edges and surfaces conflicts in the answer |

---

## 3. Architecture

```
New Paper (PDF/text)                    User Query (Streamlit chat)
         │                                        │
         ▼                                        ▼
  ┌──────────────────┐               ┌─────────────────────┐
  │ Ingestion        │               │   LangGraph Agent   │
  │ Pipeline         │               │   (orchestrator)    │
  │ (ingest.py)      │               └──────┬──────────────┘
  └──────────────────┘                      │
  Text extraction                    ┌──────┼──────────────┐
  → Chunking                         │      │              │
  → Entity extraction (Gemini)       ▼      ▼              ▼
  → Embeddings                   Vector  Graph      Contradiction
  → ArangoDB insert              Search  Traversal   Detection
         │                      (vectors) (edges)   (contradicts
         │                          │      │          edges)
         ▼                          └──────┴──────────────┘
  Knowledge Graph                              │
  grows automatically                    Merged Context
                                               │
                                               ▼
                                         Gemini Pro API
                                         (answer generation)
                                               │
                                               ▼
                                    Structured Answer
                                    + Citations
                                    + Contradictions flagged
```

### Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Knowledge Graph Store | ArangoDB (free cloud tier) | Stores nodes + edges + vector embeddings |
| LLM | Gemini Pro (your API key) | Generates final answers + entity extraction during ingestion |
| Agent Framework | LangGraph + LangChain | Orchestrates reasoning steps |
| Embeddings | Google `text-embedding-004` | Converts text to vectors for semantic search |
| PDF Parsing | PyMuPDF (`fitz`) | Extracts raw text from uploaded research PDFs |
| Text Chunking | LangChain `RecursiveCharacterTextSplitter` | Splits papers into ~500 token chunks |
| Conversation Memory | LangGraph `MemorySaver` | Persists chat history across turns in a session |
| UI | Streamlit (simple) | Demo interface — chat + paper upload |
| Language | Python 3.10+ | Everything |

---

## 4. Knowledge Graph Data Model

This is the core of the project. ArangoDB stores data as **nodes (documents)** and **edges (relationships)**.

### Node Types (Collections)

#### `studies` — Research paper summaries
```json
{
  "_key": "schoenfeld_2017",
  "title": "Dose-response relationship between weekly resistance training volume and increases in muscle mass",
  "authors": ["Brad Schoenfeld", "Dan Ogborn"],
  "year": 2017,
  "journal": "Journal of Strength and Conditioning Research",
  "summary": "Higher weekly training volumes (10+ sets per muscle) produce greater hypertrophy than lower volumes.",
  "embedding": [0.12, 0.34, ...]
}
```

#### `claims` — Specific scientific claims extracted from studies
```json
{
  "_key": "claim_rep_range_hypertrophy",
  "text": "Rep ranges of 6-30 reps are equally effective for hypertrophy when taken close to failure.",
  "confidence": "high",
  "embedding": [0.45, 0.23, ...]
}
```

#### `exercises` — Specific movements
```json
{
  "_key": "bench_press",
  "name": "Bench Press",
  "category": "compound",
  "movement_pattern": "horizontal_push",
  "embedding": [0.67, 0.12, ...]
}
```

#### `muscle_groups` — Body parts
```json
{
  "_key": "chest",
  "name": "Chest (Pectoralis Major)",
  "function": "horizontal adduction, flexion"
}
```

### Edge Types (Relationships)

| Edge Collection | From → To | Meaning |
|----------------|-----------|---------|
| `supports` | Study → Claim | This study supports this claim |
| `contradicts` | Study → Claim | This study contradicts this claim |
| `targets` | Exercise → Muscle Group | This exercise primarily targets this muscle |
| `applies_to` | Claim → Exercise | This claim applies to this exercise |
| `cites` | Study → Study | This study references this study |
| `related_to` | Claim → Claim | These claims are related |

### Why This Graph Structure Matters
When a user asks *"best exercises for chest hypertrophy"*, the agent can:
1. Find claims about hypertrophy via vector search
2. Traverse `applies_to` edges to find relevant exercises
3. Traverse `targets` edges to confirm chest involvement
4. Traverse `supports` edges back to find the original studies
5. Return a cited, structured answer — not just "do bench press"

---

## 5. Agent Design (LangGraph)

The agent is a **graph of reasoning steps** — not a linear pipeline. LangGraph lets you define nodes (actions) and edges (decisions).

### Agent Nodes

```
START
  │
  ▼
[analyze_query]        ← Classify: is this about an exercise, study, claim, or general?
  │
  ▼
[plan_retrieval]       ← Decide: vector search only? graph traversal? both?
  │
  ├──► [vector_search]       ← Semantic similarity search in ArangoDB
  │
  ├──► [graph_traversal]     ← Follow edges in the knowledge graph
  │
  └──► [hybrid_search]       ← Both, then merge results
          │
          ▼
     [synthesize]            ← Merge retrieved context, remove duplicates
          │
          ▼
     [generate_answer]       ← Send context + query to Gemini, get answer
          │
          ▼
     [format_response]       ← Add citations, structure the output
          │
         END
```

### Agent State (what the agent tracks between steps)
```python
class AgentState(TypedDict):
    query: str
    query_type: str           # "exercise" | "study" | "claim" | "general"
    retrieval_plan: str       # "vector" | "graph" | "hybrid"
    vector_results: list
    graph_results: list
    merged_context: str
    final_answer: str
    citations: list
```

---

## 6. RAG Pipeline (The Foundation — Build This First)

Before building the agent, you build a standalone RAG pipeline. This teaches you the basics.

### Step 1 — Ingestion
```
Raw data (JSON fitness research summaries)
  → Generate embeddings (Gemini text-embedding-004)
  → Store in ArangoDB (node + embedding field)
```

### Step 2 — Retrieval
```
User query
  → Generate query embedding
  → ArangoDB vector similarity search (cosine similarity)
  → Return top-K matching nodes
```

### Step 3 — Generation
```
Retrieved chunks + original query
  → Prompt template
  → Gemini Pro
  → Answer
```

### What Makes It "Hybrid RAG"
Standard RAG: vector search only → flat text chunks → answer
Hybrid RAG (this project): vector search + graph traversal → structured, related entities → richer answer

---

## 6A. Dynamic Ingestion Pipeline (Growing Knowledge Base)

This is what separates a toy project from a real system. Instead of manually writing JSON, you drop a PDF of a new research paper and the pipeline automatically extracts, chunks, embeds, and inserts it into the knowledge graph.

### Pipeline Steps

```
User uploads PDF (via Streamlit sidebar or CLI)
         │
         ▼
[1] Text Extraction (PyMuPDF)
    → raw text from the PDF
         │
         ▼
[2] Chunking (LangChain RecursiveCharacterTextSplitter)
    → ~500 token chunks with 50 token overlap
    → overlap ensures no sentence is cut off between chunks
         │
         ▼
[3] Entity Extraction (Gemini Pro)
    → prompt: "From this text, extract:
               - Paper title and authors
               - Year and journal
               - Key scientific claims made
               - Exercises or muscle groups mentioned"
    → returns structured JSON
         │
         ▼
[4] Embedding Generation (text-embedding-004)
    → embed each chunk AND each extracted claim
         │
         ▼
[5] ArangoDB Insertion
    → insert Study node (if new paper)
    → insert Claim nodes for each extracted claim
    → insert edges: Study→supports→Claim
    → insert edges: Claim→applies_to→Exercise (if mentioned)
    → insert chunk embeddings for vector search
         │
         ▼
Knowledge graph updated — agent can now answer questions using the new paper
```

### Why Chunking Matters
A research paper is too long to embed as one unit. A single embedding for 5000 words loses all the detail. Chunking breaks it into smaller pieces so each piece gets its own precise embedding — and vector search finds the *specific* relevant paragraph, not the whole paper.

### Entity Extraction Prompt (Example)
```python
EXTRACTION_PROMPT = """
You are a fitness science knowledge extractor.
Given the following research paper text, extract structured information.

Text: {chunk_text}

Return a JSON object with:
- "claims": list of specific scientific claims made (each as a string)
- "exercises": list of exercises mentioned (empty list if none)
- "muscle_groups": list of muscle groups mentioned (empty list if none)
- "supports_claims": list of claims this paper supports
- "contradicts_claims": list of claims this paper contradicts (if any)

Return ONLY valid JSON, no other text.
"""
```

### New File Added
```
src/
└── ingestion_pipeline.py   ← the full pipeline above as one callable function
```

---

## 6B. Contradiction Detection Node

When a user asks a question, the agent doesn't just retrieve supporting evidence — it also checks if there are **studies in the graph that contradict** the retrieved claims. This is unique, genuinely useful, and directly shows graph reasoning in action.

### How It Works

```
After hybrid retrieval:
         │
         ▼
[contradiction_check node]
    → for each retrieved Claim node
    → run AQL query: find Studies connected via `contradicts` edges
    → if contradicting studies found → add to context with a flag
         │
         ▼
[generate_answer]
    → Gemini prompt includes both supporting AND contradicting evidence
    → Answer includes a "Note: conflicting evidence" section if contradictions exist
```

### Example Output
```
User: "Is high volume training always better for hypertrophy?"

Agent answer:
"Most evidence supports higher training volumes (10-20 sets/muscle/week)
for hypertrophy (Schoenfeld 2017, Krieger 2010).

⚠️ Conflicting evidence: Ralston et al. 2017 found no significant
difference between low and high volume when intensity was equated.
Newer meta-analyses suggest diminishing returns beyond 20 sets/week
for natural lifters.

Recommendation: 10-15 sets/muscle/week is a safe evidence-based
starting point, with individual variation."
```

This is far more useful than a system that just agrees with whatever it finds first.

### Updated Agent State
```python
class AgentState(TypedDict):
    query: str
    query_type: str
    retrieval_plan: str
    vector_results: list
    graph_results: list
    contradictions: list        # NEW — conflicting studies found
    merged_context: str
    final_answer: str
    citations: list
    chat_history: list          # NEW — conversation memory
```

### Updated Agent Nodes
```
START
  │
  ▼
[analyze_query]
  │
  ▼
[plan_retrieval]
  │
  ├──► [vector_search]
  ├──► [graph_traversal]
  └──► [hybrid_search]
          │
          ▼
     [contradiction_check]    ← NEW: check for conflicting studies
          │
          ▼
     [synthesize]
          │
          ▼
     [generate_answer]        ← now includes contradictions in prompt
          │
          ▼
     [format_response]
          │
         END
```

---

## 7. Sample Data (You Will Create This)

You do NOT need to scrape or parse real PDFs. You will create ~15-20 curated JSON entries covering:

**Studies to include (summaries, not full papers):**
- Schoenfeld 2017 — training volume and hypertrophy
- Krieger 2010 — sets per exercise and muscle growth
- Helms 2014 — rep ranges for natural athletes
- Schoenfeld 2016 — training frequency for hypertrophy
- Morton 2016 — protein synthesis and rep ranges

**Claims to include:**
- Optimal rep range for hypertrophy (6-30 reps near failure)
- Progressive overload is the primary driver of muscle growth
- Higher training frequency (2x/week per muscle) outperforms 1x
- 10-20 sets per muscle per week is the effective hypertrophy range
- Protein intake 1.6-2.2g/kg is sufficient for muscle growth

**Exercises:** Bench Press, Squat, Deadlift, Pull-up, Row, Overhead Press, Dumbbell Fly, Leg Press, RDL, Hip Thrust

**Muscle Groups:** Chest, Back, Legs (Quads/Hamstrings/Glutes), Shoulders, Biceps, Triceps

---

## 8. Tech Stack & Setup

### Dependencies
```
python >= 3.10
langchain
langchain-google-genai
langgraph
python-arango          # ArangoDB Python driver
pymupdf                # PDF text extraction (imported as fitz)
streamlit              # UI
python-dotenv          # API key management
```

### ArangoDB Setup
- Use **ArangoDB Cloud (ArangoGraph)** — free tier available
- Create a free cluster at cloud.arangodb.com
- You get a managed ArangoDB instance, no local install needed

### Project Folder Structure
```
fitscience/
├── .env                        # API keys (never commit this)
├── requirements.txt
├── data/
│   ├── fitness_knowledge.json  # Seed data (studies, claims, exercises)
│   └── papers/                 # Drop new PDFs here for ingestion
├── src/
│   ├── ingest.py               # Seed data loader (Day 1)
│   ├── ingestion_pipeline.py   # Dynamic PDF → graph pipeline (Day 2)
│   ├── rag.py                  # Basic RAG pipeline (Day 1)
│   ├── graph_retriever.py      # Graph traversal + contradiction queries (Day 2)
│   ├── agent.py                # LangGraph agent with all nodes (Day 3)
│   └── prompts.py              # All prompt templates
├── app.py                      # Streamlit UI — chat + PDF upload sidebar
└── notebooks/
    └── exploration.ipynb       # For testing and learning each piece
```

---

## 9. Build Order (Day-by-Day Plan)

### Day 1 — Foundation: Basic RAG (3-4 hours)
**Goal: Understand RAG, get seed data into ArangoDB**

1. Set up ArangoDB Cloud account (free tier) + create database + collections
2. Install all dependencies, set up `.env` with Gemini API key
3. Create `fitness_knowledge.json` — write 15-20 seed entries (studies, claims, exercises, muscle groups)
4. Build `ingest.py` — load seed nodes into ArangoDB, generate and store embeddings
5. Build `rag.py` — basic vector search query + Gemini answer generation
6. Test in notebook: ask 3-4 questions, verify answers cite real seed data

**By end of Day 1 you understand:** What RAG is, what embeddings are, how vector similarity search works

---

### Day 2 — Graph + Hybrid RAG + Ingestion Pipeline (4-5 hours)
**Goal: Add graph relationships, contradiction detection, and dynamic ingestion**

1. Extend `ingest.py` to create edge collections and insert all edges (supports, contradicts, targets, applies_to)
2. Build `graph_retriever.py` — AQL graph traversal queries + contradiction lookup queries
3. Test graph traversal in notebook: given a muscle group → find exercises → find supporting studies
4. Build `ingestion_pipeline.py` — PDF text extraction → chunking → Gemini entity extraction → ArangoDB insert
5. Test pipeline: drop one new research PDF, verify it appears in ArangoDB correctly
6. Build hybrid retrieval function: merge vector search + graph traversal results

**By end of Day 2 you understand:** Knowledge graphs, GraphRAG, why dynamic ingestion matters, what chunking solves

---

### Day 3 — LangGraph Agent + UI (3-4 hours)
**Goal: Wrap everything in an intelligent agent, make it demo-ready**

1. Build `agent.py` with LangGraph — define all nodes including `contradiction_check`
2. Add `MemorySaver` for conversation memory (multi-turn chat)
3. Wire all retrieval functions as agent nodes with conditional routing
4. Build `app.py` — Streamlit UI with:
   - Chat interface (left panel)
   - PDF upload sidebar (right panel) that triggers ingestion pipeline
5. Test full flow end-to-end: chat → ask question → upload new paper → ask again using new paper
6. Prepare 4 demo questions (see Section 10)

**By end of Day 3 you have:** A fully working intermediate-level demo for the interview

---

## 10. Interview Talking Points

When asked "tell me about a project you built," walk through this narrative:

1. **Problem:** "Most fitness advice online is anecdotal. I wanted to build something that reasons over actual peer-reviewed research."

2. **Why a knowledge graph:** "Flat vector search treats every document the same. But research has structure — a study *supports* a claim, a claim *applies to* an exercise, an exercise *targets* a muscle. I needed to model those relationships, not just similarity."

3. **Why ArangoDB:** "ArangoDB lets me store both the vector embeddings and the graph relationships in one database, and query them together. I don't need a separate graph DB and a separate vector store."

4. **What an agent adds:** "A simple RAG pipeline always does the same thing. My LangGraph agent *decides* — for some queries it only needs vector search, for others it traverses the graph, for complex questions it does both. That's the difference between a pipeline and an agent."

5. **What makes it not a toy:** "The knowledge base isn't static. I built a PDF ingestion pipeline — you drop a new research paper in, it gets chunked, entities are extracted via Gemini, embeddings are generated, and the graph updates automatically. The agent can use that paper in the next query."

6. **Contradiction detection:** "Most RAG systems just find supporting evidence. My agent also checks for contradicting studies using the `contradicts` edges in the graph. If two studies disagree, the answer flags that conflict rather than hiding it. That's closer to how a researcher actually thinks."

7. **What I'd do next:** "Add PubMed API integration so users can ingest papers by DOI, add a user profile node (training experience, goals) so the agent personalizes recommendations, and add an evaluation layer that scores answer quality."

---

## 11. Key Concepts Explained Simply

### What is an Agent?
A regular program follows a fixed sequence of steps. An **agent** observes its environment, decides what action to take next, takes that action, observes the result, and decides again. It has a *reasoning loop*. LangGraph lets you build this loop as a graph of nodes and edges.

### What is RAG?
**Retrieval Augmented Generation.** Instead of asking an LLM a question cold (it might hallucinate), you first *retrieve* relevant documents from your own database, then *augment* the LLM's prompt with that retrieved context. The LLM *generates* an answer grounded in your data.

### What is GraphRAG?
Standard RAG retrieves flat text chunks by similarity. **GraphRAG** retrieves *structured knowledge* by traversing relationships in a graph. It finds not just "similar documents" but "related entities connected through typed edges." The answers are richer because the context is richer.

### What is Hybrid RAG?
Combining two or more retrieval strategies. In this project: **vector search** (find semantically similar content) + **graph traversal** (follow typed relationships). Each catches what the other misses.

---

## 12. Resources to Study Alongside Building

| Topic | Resource |
|-------|----------|
| LangGraph basics | LangGraph docs — "Build a basic chatbot" + "How to add memory" tutorials |
| LangGraph agent with tools | LangGraph docs — "Build an agent with tools" tutorial |
| ArangoDB + Python | python-arango docs + ArangoDB AQL tutorial (graph traversal section) |
| ArangoDB vector search | ArangoDB docs — "Vector search" section |
| RAG concepts | LangChain RAG tutorial (official docs) |
| Text chunking | LangChain docs — `RecursiveCharacterTextSplitter` |
| GraphRAG paper | Microsoft GraphRAG blog post (accessible, non-math heavy) |
| Gemini API in LangChain | `langchain-google-genai` README on GitHub |
| PyMuPDF (PDF parsing) | PyMuPDF docs — "Tutorial" section |

---

*This document is your source of truth. Implementation starts after reviewing this.*
