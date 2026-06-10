"""ArangoDB connection and schema setup for the FitScience knowledge graph."""
import os

from arango import ArangoClient
from dotenv import load_dotenv

load_dotenv()

NODE_COLLECTIONS = ["studies", "claims", "exercises", "muscle_groups", "chunks"]
EDGE_COLLECTIONS = ["supports", "contradicts", "cites", "applies_to", "targets", "has_chunk"]


def get_db():
    """Connect to the fitscience database using credentials from .env."""
    # Required in .env: ARANGO_URL, ARANGO_USER, ARANGO_PASSWORD. ARANGO_DB defaults to "fitscience".
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
