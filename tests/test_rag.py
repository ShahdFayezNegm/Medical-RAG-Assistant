import sys
from pathlib import Path


sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
    )
)


from src.rag_pipeline import MedicalRAGPipeline


rag = MedicalRAGPipeline()


question = (
    "What is the first-line "
    "treatment for type 2 diabetes?"
)


result = rag.ask(
    question
)

print("=" * 60)
print("RAG TEST")
print("=" * 60)

print()
print("Question:")
print(question)

print()
print("Answer:")
print(result["answer"])

print()
print("Sources:")

for source in result.get(
    "sources",
    []
):

    print(
        f"- {source['source']} "
        f"| Page {source['page']}"
    )

print()
print("Evidence:")

for item in result.get(
    "evidence",
    []
):

    print(
        f"- Document {item['document']} "
        f"| {item['source']} "
        f"| Page {item['page']}"
    )