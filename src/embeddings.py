"""Gemini embedding helpers. One place to change the model later."""
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

EMBED_MODEL = "models/text-embedding-004"

_embedder = None


def _get_embedder():
    """Return the singleton embedder, creating it on first call."""
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
