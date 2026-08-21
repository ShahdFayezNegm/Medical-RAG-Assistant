import sys
from pathlib import Path
import time

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.rag_pipeline import MedicalRAGPipeline


print("=" * 70)
print("BUILDING MEDICAL RAG PIPELINE")
print("=" * 70)

rag = MedicalRAGPipeline()

questions = [
    "What is the first-line treatment for type 2 diabetes?",
    "What are the treatment options for type 2 diabetes?",
    "What is recommended for adults with type 2 diabetes?",
    "What is the treatment for hypertension?",
    "What is step 1 treatment for hypertension?",
]

print()
print("=" * 70)
print("MULTIPLE RAG TEST")
print("=" * 70)

for i, question in enumerate(questions, 1):

    print()
    print("=" * 70)
    print(f"QUESTION {i}")
    print("=" * 70)

    print()
    print("Question:")
    print(question)

    start = time.time()

    result = rag.ask(question)

    elapsed = time.time() - start

    print()
    print("Answer:")
    print(result["answer"])

    print()
    print("Sources:")

    for source in result.get("sources", []):
        print(
            f"- {source['source']} "
            f"| Page {source['page']}"
        )

    print()
    print("Evidence:")

    for item in result.get("evidence", []):
        print(
            f"- Document {item['document']} "
            f"| {item['source']} "
            f"| Page {item['page']}"
        )

    print()
    print(f"Time: {elapsed:.2f} seconds")


print()
print("=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)