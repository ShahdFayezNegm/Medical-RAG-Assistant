import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from src.rag_pipeline import MedicalRAGPipeline


# ==========================================================
# Evaluation Dataset
# ==========================================================

TEST_CASES = [

    {
        "question": "What is the first-line treatment for type 2 diabetes?",
        "expected_source": "diabetes_guideline.pdf",
        "expected_page": 68
    },

    {
        "question": "What are the symptoms of diabetes?",
        "expected_source": "diabetes_guideline.pdf",
        "expected_page": 115
    },

    {
        "question": "What are the risk factors for hypertension?",
        "expected_source": "hypertension_guideline.pdf",
        "expected_page": 9
    },

    {
        "question": "What is the recommended treatment for hypertension?",
        "expected_source": "hypertension_guideline.pdf",
        "expected_page": 21
    },

    {
        "question": "What is the treatment for a disease that is not mentioned in the documents?",
        "expected_source": None,
        "expected_page": None
    }
]


# ==========================================================
# Metrics
# ==========================================================

def calculate_rank(documents, expected_source, expected_page):

    for rank, doc in enumerate(documents, start=1):

        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", None)

        source_match = (
            expected_source is not None
            and expected_source in source
        )

        page_match = (
            expected_page is not None
            and str(page) == str(expected_page)
        )

        if source_match and page_match:
            return rank

    return None


def reciprocal_rank(rank):

    if rank is None:
        return 0.0

    return 1.0 / rank


# ==========================================================
# Build Pipeline
# ==========================================================

print("=" * 60)
print("Building Medical RAG Pipeline...")
print("=" * 60)

rag = MedicalRAGPipeline()


# ==========================================================
# Evaluation
# ==========================================================

hit_at_1 = 0
hit_at_3 = 0
hit_at_5 = 0

mrr_scores = []

unknown_pass = 0
unknown_total = 0


for i, test in enumerate(TEST_CASES, start=1):

    question = test["question"]

    print("\n" + "=" * 60)
    print(f"TEST {i}")
    print("=" * 60)

    print("\nQuestion:")
    print(question)

    # ------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------

    docs = rag.retriever.search(
        question,
        k=5
    )

    print("\nTop 5 Retrieved Documents:")

    for rank, doc in enumerate(docs, start=1):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        print(
            f"{rank}. {source} | Page {page}"
        )

    # ------------------------------------------------------
    # Unknown Question
    # ------------------------------------------------------

    if test["expected_source"] is None:

        unknown_total += 1

        result = rag.generator.generate(
            question,
            docs
        )

        answer = result["answer"]

        print("\nAnswer:")
        print(answer)

        if "I don't have enough information" in answer:

            unknown_pass += 1

            print("\nUnknown Question Rejection: PASS")

        else:

            print("\nUnknown Question Rejection: FAIL")

        continue

    # ------------------------------------------------------
    # Rank
    # ------------------------------------------------------

    rank = calculate_rank(
        docs,
        test["expected_source"],
        test["expected_page"]
    )

    print("\nCorrect Page Rank:")

    if rank is not None:
        print(f"#{rank}")
    else:
        print("Not found")

    # ------------------------------------------------------
    # Hit@K
    # ------------------------------------------------------

    if rank == 1:
        hit_at_1 += 1

    if rank is not None and rank <= 3:
        hit_at_3 += 1

    if rank is not None and rank <= 5:
        hit_at_5 += 1

    # ------------------------------------------------------
    # MRR
    # ------------------------------------------------------

    mrr_scores.append(
        reciprocal_rank(rank)
    )

    # ------------------------------------------------------
    # Generate Answer
    # ------------------------------------------------------

    result = rag.generator.generate(
        question,
        docs
    )

    print("\nAnswer:")
    print(result["answer"])


# ==========================================================
# Final Metrics
# ==========================================================

known_total = len(TEST_CASES) - unknown_total

mrr = (
    sum(mrr_scores) / known_total
    if known_total > 0
    else 0
)

hit1_rate = (
    hit_at_1 / known_total
    if known_total > 0
    else 0
)

hit3_rate = (
    hit_at_3 / known_total
    if known_total > 0
    else 0
)

hit5_rate = (
    hit_at_5 / known_total
    if known_total > 0
    else 0
)

unknown_rate = (
    unknown_pass / unknown_total
    if unknown_total > 0
    else 0
)


# ==========================================================
# Summary
# ==========================================================

print("\n")
print("#" * 60)
print("FINAL RETRIEVAL EVALUATION")
print("#" * 60)

print(
    f"Hit@1: {hit_at_1}/{known_total} "
    f"({hit1_rate:.2%})"
)

print(
    f"Hit@3: {hit_at_3}/{known_total} "
    f"({hit3_rate:.2%})"
)

print(
    f"Hit@5: {hit_at_5}/{known_total} "
    f"({hit5_rate:.2%})"
)

print(
    f"MRR: {mrr:.3f}"
)

print(
    f"Unknown Rejection: "
    f"{unknown_pass}/{unknown_total} "
    f"({unknown_rate:.2%})"
)

print("#" * 60)
