import sys
from pathlib import Path


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.loader import PDFLoader
from src.bm25_retriever import BM25Retriever


# =========================================================
# Load Documents
# =========================================================

loader = PDFLoader("data")

documents = loader.load_documents()


# =========================================================
# Build BM25
# =========================================================

retriever = BM25Retriever(
    documents
)


# =========================================================
# Query
# =========================================================

query = "metformin"

results = retriever.search(
    query,
    k=5
)


# =========================================================
# Results
# =========================================================

print("=" * 60)
print("BM25 TEST")
print("=" * 60)

print(
    "Query:",
    query
)

print(
    "Documents:",
    len(documents)
)

print()

for i, doc in enumerate(
    results,
    start=1
):

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
    "BM25 test completed."
)