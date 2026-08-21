import sys
from pathlib import Path


# =========================================================
# Add project root to Python path
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


# =========================================================
# Imports
# =========================================================

from src.loader import PDFLoader
from src.splitter import DocumentSplitter
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever


# =========================================================
# Load documents
# =========================================================

loader = PDFLoader("data")

documents = loader.load_documents()


# =========================================================
# Split documents
# =========================================================

splitter = DocumentSplitter()

chunks = splitter.split_documents(
    documents
)


# =========================================================
# Embeddings
# =========================================================

embedding = (
    EmbeddingModel()
    .get_embeddings()
)


# =========================================================
# FAISS
# =========================================================

vector_store = VectorStore(
    embedding
)

db = vector_store.create(
    chunks
)


# =========================================================
# Retriever
# =========================================================

retriever = Retriever(
    db
)


# =========================================================
# Query
# =========================================================

query = "HbA1c target"

results = retriever.mmr_search(
    query=query,
    k=5,
    fetch_k=15
)


# =========================================================
# Results
# =========================================================

print("=" * 60)
print("MMR TEST")
print("=" * 60)

print(
    "Query:",
    query
)

print(
    "Documents:",
    len(documents)
)

print(
    "Chunks:",
    len(chunks)
)

print(
    "Retrieved:",
    len(results)
)

for i, doc in enumerate(
    results,
    start=1
):

    print()
    print(
        f"Result {i}"
    )

    print("-" * 40)

    print(
        "Source:",
        doc.metadata.get(
            "source",
            "Unknown"
        )
    )

    print(
        "Page:",
        doc.metadata.get(
            "page",
            "Unknown"
        )
    )

    print()

    print(
        doc.page_content[:500]
    )


print()
print(
    "MMR test completed."
)