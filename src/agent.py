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
