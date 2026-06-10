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
