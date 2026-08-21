import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.loader import PDFLoader


loader = PDFLoader("data")

docs = loader.load_documents()

print("=" * 60)
print("LOADER TEST")
print("=" * 60)

print("Loaded pages:", len(docs))

for i, doc in enumerate(docs[:10], start=1):

    print()
    print(f"Page {i}")
    print("-" * 40)

    print(
        "Source:",
        doc.metadata.get("source", "Unknown")
    )

    print(
        "Page:",
        doc.metadata.get("page", "Unknown")
    )

    print(
        doc.page_content[:500]
    )

print()
print("Loader test completed.")