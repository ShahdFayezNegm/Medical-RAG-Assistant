import sys
from pathlib import Path


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.loader import PDFLoader
from src.splitter import DocumentSplitter


# =========================================================
# Load Documents
# =========================================================

loader = PDFLoader("data")

documents = loader.load_documents()


# =========================================================
# Split Documents
# =========================================================

splitter = DocumentSplitter()

chunks = splitter.split_documents(
    documents
)


# =========================================================
# Results
# =========================================================

print("=" * 60)
print("SPLITTER TEST")
print("=" * 60)

print(
    "Documents:",
    len(documents)
)

print(
    "Chunks:",
    len(chunks)
)


if chunks:

    print()
    print("First Chunk")
    print("-" * 40)

    print(
        chunks[0].page_content
    )

    print()
    print("Metadata")
    print("-" * 40)

    print(
        chunks[0].metadata
    )


print()
print("Splitter test completed.")