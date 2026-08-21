import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.rag_pipeline import MedicalRAGPipeline


# ---------------------------------------------------------
# Change this to an actual PDF path for testing
# ---------------------------------------------------------

TEST_PDF = r"C:\Users\CS\Desktop\Medical-RAG-Assistant\data\HIS7_HeartAttack_HeartCond_Booklet.pdf"


# ---------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------

rag = MedicalRAGPipeline()


# ---------------------------------------------------------
# Add PDF
# ---------------------------------------------------------

result = rag.add_pdf(
    TEST_PDF
)


# ---------------------------------------------------------
# Print result
# ---------------------------------------------------------

print()
print("=" * 60)
print("ADD PDF TEST")
print("=" * 60)

print(
    "Filename:",
    result["filename"]
)

print(
    "Pages added:",
    result["pages_added"]
)

print(
    "Chunks added:",
    result["chunks_added"]
)

print(
    "Total documents:",
    result["total_documents"]
)

print(
    "Total chunks:",
    result["total_chunks"]
)