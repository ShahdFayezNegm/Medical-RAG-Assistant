import sys
from pathlib import Path


# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


# ============================================================
# Imports
# ============================================================

from src.rag_pipeline import MedicalRAGPipeline


# ============================================================
# Test Cases
# ============================================================

TEST_CASES = [
    {
        "name": "TEST 1",
        "question": "What is the first-line treatment for type 2 diabetes?",
        "expected_source": "diabetes_guideline.pdf",
        "expected_page": 68,
        "important_pages": [68, 130],
    },

    {
        "name": "TEST 2",
        "question": "What are the symptoms of diabetes?",
        "expected_source": "diabetes_guideline.pdf",
        "expected_page": 115,
        "important_pages": [115, 116],
    },

    {
        "name": "TEST 3",
        "question": "What are the risk factors for hypertension?",
        "expected_source": "hypertension_guideline.pdf",
        "expected_page": 9,
        "important_pages": [9, 47],
    },

    {
        "name": "TEST 4",
        "question": "What is the recommended treatment for hypertension?",
        "expected_source": "hypertension_guideline.pdf",
        "expected_page": 21,
        "important_pages": [21, 41],
    },
]


# ============================================================
# Helper: page rank
# ============================================================

def find_rank(
    documents,
    expected_source,
    expected_page
):
    """
    Return the 1-based rank of the expected page.
    """

    for rank, doc in enumerate(
        documents,
        start=1
    ):

        source = str(
            doc.metadata.get(
                "source",
                ""
            )
        ).lower()

        page = str(
            doc.metadata.get(
                "page",
                ""
            )
        )

        if (
            expected_source.lower() in source
            and page == str(expected_page)
        ):
            return rank

    return None


# ============================================================
# Helper: print document
# ============================================================

def print_document(
    doc,
    label,
    max_chars=1800
):
    """
    Print useful information about one document/chunk.
    """

    source = doc.metadata.get(
        "source",
        "Unknown"
    )

    page = doc.metadata.get(
        "page",
        "Unknown"
    )

    print()
    print("=" * 70)
    print(label)
    print("=" * 70)

    print(
        f"Source: {source}"
    )

    print(
        f"Page: {page}"
    )

    print(
        f"FAISS rank: "
        f"{doc.metadata.get('faiss_rank')}"
    )

    print(
        f"BM25 rank: "
        f"{doc.metadata.get('bm25_rank')}"
    )

    print(
        f"CrossEncoder score: "
        f"{doc.metadata.get('cross_encoder_score')}"
    )

    print(
        f"Fusion score: "
        f"{doc.metadata.get('fusion_score')}"
    )

    print()
    print("CONTENT:")
    print("-" * 70)

    content = doc.page_content.strip()

    if len(content) > max_chars:
        print(
            content[:max_chars]
        )

        print()
        print(
            f"... [truncated, "
            f"showing first {max_chars} characters]"
        )

    else:
        print(content)


# ============================================================
# Helper: find all docs for a page
# ============================================================

def find_page_documents(
    documents,
    source,
    page
):
    """
    Find all chunks belonging to a specific PDF page.
    """

    matches = []

    for doc in documents:

        doc_source = str(
            doc.metadata.get(
                "source",
                ""
            )
        ).lower()

        doc_page = str(
            doc.metadata.get(
                "page",
                ""
            )
        )

        if (
            source.lower() in doc_source
            and doc_page == str(page)
        ):
            matches.append(doc)

    return matches


# ============================================================
# Build pipeline
# ============================================================

print("=" * 70)
print("BUILDING MEDICAL RAG PIPELINE")
print("=" * 70)

rag = MedicalRAGPipeline()

print()
print(
    "Pipeline ready."
)


# ============================================================
# Global counters
# ============================================================

hit_at_1 = 0
hit_at_3 = 0
hit_at_5 = 0

ranks = []


# ============================================================
# Run tests
# ============================================================

for case in TEST_CASES:

    name = case["name"]
    question = case["question"]

    expected_source = case[
        "expected_source"
    ]

    expected_page = case[
        "expected_page"
    ]

    important_pages = case[
        "important_pages"
    ]

    print()
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print()
    print("QUESTION:")
    print(question)

    # ========================================================
    # 1. FAISS / MMR
    # ========================================================

    semantic_results = (
        rag.retriever.semantic.mmr_search(
            query=question,
            k=50,
            fetch_k=150
        )
    )

    semantic_rank = find_rank(
        semantic_results,
        expected_source,
        expected_page
    )

    print()
    print(
        "FAISS / MMR"
    )

    print(
        f"Candidates: "
        f"{len(semantic_results)}"
    )

    print(
        f"Expected Page Rank: "
        f"{semantic_rank}"
    )

    # ========================================================
    # 2. BM25
    # ========================================================

    keyword_results = (
        rag.retriever.keyword.search(
            query=question,
            k=50
        )
    )

    keyword_rank = find_rank(
        keyword_results,
        expected_source,
        expected_page
    )

    print()
    print(
        "BM25"
    )

    print(
        f"Candidates: "
        f"{len(keyword_results)}"
    )

    print(
        f"Expected Page Rank: "
        f"{keyword_rank}"
    )

    # ========================================================
    # 3. Build merged candidate pool
    # ========================================================

    merged = []
    seen = set()

    semantic_rank_map = {}
    keyword_rank_map = {}

    # --------------------------------------------------------
    # FAISS ranks
    # --------------------------------------------------------

    for rank, doc in enumerate(
        semantic_results,
        start=1
    ):

        key = (
            doc.metadata.get(
                "source",
                ""
            ),
            doc.metadata.get(
                "page",
                ""
            ),
            doc.page_content.strip()
        )

        semantic_rank_map[key] = rank

    # --------------------------------------------------------
    # BM25 ranks
    # --------------------------------------------------------

    for rank, doc in enumerate(
        keyword_results,
        start=1
    ):

        key = (
            doc.metadata.get(
                "source",
                ""
            ),
            doc.metadata.get(
                "page",
                ""
            ),
            doc.page_content.strip()
        )

        keyword_rank_map[key] = rank

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    for doc in (
        semantic_results
        + keyword_results
    ):

        key = (
            doc.metadata.get(
                "source",
                ""
            ),
            doc.metadata.get(
                "page",
                ""
            ),
            doc.page_content.strip()
        )

        if key in seen:
            continue

        seen.add(key)

        doc.metadata = dict(
            doc.metadata
        )

        doc.metadata[
            "faiss_rank"
        ] = semantic_rank_map.get(
            key
        )

        doc.metadata[
            "bm25_rank"
        ] = keyword_rank_map.get(
            key
        )

        merged.append(doc)

    print()
    print(
        "HYBRID BEFORE RERANKING"
    )

    print(
        f"Merged candidates: "
        f"{len(merged)}"
    )

    hybrid_rank = find_rank(
        merged,
        expected_source,
        expected_page
    )

    print(
        f"Expected Page Rank: "
        f"{hybrid_rank}"
    )

    # ========================================================
    # 4. Source filtering
    # ========================================================

    question_lower = (
        question.lower()
    )

    target_source = None

    if (
        "hypertension" in question_lower
        or "high blood pressure"
        in question_lower
        or "blood pressure"
        in question_lower
    ):
        target_source = (
            "hypertension_guideline.pdf"
        )

    elif (
        "diabetes" in question_lower
        or "t2dm" in question_lower
    ):
        target_source = (
            "diabetes_guideline.pdf"
        )

    if target_source:

        filtered = [
            doc
            for doc in merged
            if target_source.lower()
            in doc.metadata.get(
                "source",
                ""
            ).lower()
        ]

        if len(filtered) >= 5:

            merged = filtered

            print()
            print(
                f"Target source detected: "
                f"{target_source}"
            )

            print(
                f"Candidates after "
                f"source filtering: "
                f"{len(merged)}"
            )

    # ========================================================
    # 5. Reranking
    # ========================================================

    reranked = (
        rag.retriever.reranker.rerank(
            query=question,
            documents=merged,
            top_k=5
        )
    )

    final_rank = find_rank(
        reranked,
        expected_source,
        expected_page
    )

    print()
    print(
        "FINAL HYBRID + RERANKER"
    )

    print(
        f"Expected Page Rank: "
        f"{final_rank}"
    )

    # ========================================================
    # 6. Final results
    # ========================================================

    print()
    print(
        "TOP 5 RESULTS"
    )

    print("-" * 70)

    for rank, doc in enumerate(
        reranked,
        start=1
    ):

        print(
            f"{rank}. "
            f"{doc.metadata.get('source', 'Unknown')} "
            f"| Page "
            f"{doc.metadata.get('page', 'Unknown')}"
        )

        print(
            f"   FAISS: "
            f"{doc.metadata.get('faiss_rank')}"
        )

        print(
            f"   BM25: "
            f"{doc.metadata.get('bm25_rank')}"
        )

        print(
            f"   CrossEncoder: "
            f"{doc.metadata.get('cross_encoder_score')}"
        )

        print(
            f"   Fusion: "
            f"{doc.metadata.get('fusion_score')}"
        )

    # ========================================================
    # 7. Update metrics
    # ========================================================

    if final_rank == 1:
        hit_at_1 += 1

    if final_rank is not None and final_rank <= 3:
        hit_at_3 += 1

    if final_rank is not None and final_rank <= 5:
        hit_at_5 += 1

    if final_rank is not None:
        ranks.append(
            final_rank
        )

    # ========================================================
    # 8. IMPORTANT PAGE INSPECTION
    # ========================================================

    print()
    print("=" * 70)
    print(
        "IMPORTANT PAGE INSPECTION"
    )
    print("=" * 70)

    print(
        "These pages are printed to diagnose "
        "hard negatives and the expected page."
    )

    # --------------------------------------------------------
    # First search in final results
    # --------------------------------------------------------

    printed_keys = set()

    for page in important_pages:

        found = False

        for doc in reranked:

            source = str(
                doc.metadata.get(
                    "source",
                    ""
                )
            ).lower()

            doc_page = str(
                doc.metadata.get(
                    "page",
                    ""
                )
            )

            if (
                expected_source.lower()
                in source
                and doc_page == str(page)
            ):

                key = (
                    source,
                    doc_page,
                    doc.page_content[:100]
                )

                if key not in printed_keys:

                    print_document(
                        doc,
                        f"IMPORTANT PAGE {page}"
                    )

                    printed_keys.add(
                        key
                    )

                found = True

        # ----------------------------------------------------
        # If not in final results, search merged candidates
        # ----------------------------------------------------

        if not found:

            page_docs = (
                find_page_documents(
                    merged,
                    expected_source,
                    page
                )
            )

            for doc in page_docs:

                key = (
                    doc.metadata.get(
                        "source",
                        ""
                    ),
                    doc.metadata.get(
                        "page",
                        ""
                    ),
                    doc.page_content[:100]
                )

                if key in printed_keys:
                    continue

                # Manually calculate CrossEncoder score
                # for diagnostic purposes
                score = (
                    rag.retriever.reranker.score(
                        question,
                        [doc]
                    )[0]
                )

                doc.metadata = dict(
                    doc.metadata
                )

                doc.metadata[
                    "cross_encoder_score"
                ] = score

                print_document(
                    doc,
                    f"IMPORTANT PAGE {page} "
                    f"(candidate, not Top-5)"
                )

                printed_keys.add(
                    key
                )

    # ========================================================
    # 9. Print expected page status
    # ========================================================

    print()
    print(
        "EXPECTED PAGE STATUS"
    )
    print("-" * 70)

    print(
        f"Expected source: "
        f"{expected_source}"
    )

    print(
        f"Expected page: "
        f"{expected_page}"
    )

    print(
        f"Final rank: "
        f"{final_rank}"
    )

    if final_rank == 1:

        print(
            "STATUS: PASS @1"
        )

    elif (
        final_rank is not None
        and final_rank <= 3
    ):

        print(
            "STATUS: PASS @3"
        )

    elif (
        final_rank is not None
        and final_rank <= 5
    ):

        print(
            "STATUS: PASS @5"
        )

    else:

        print(
            "STATUS: FAIL"
        )


# ============================================================
# Final evaluation summary
# ============================================================

print()
print()
print("=" * 70)
print(
    "FINAL RETRIEVAL SUMMARY"
)
print("=" * 70)

total = len(
    TEST_CASES
)

print(
    f"Hit@1: "
    f"{hit_at_1}/{total} "
    f"({100 * hit_at_1 / total:.2f}%)"
)

print(
    f"Hit@3: "
    f"{hit_at_3}/{total} "
    f"({100 * hit_at_3 / total:.2f}%)"
)

print(
    f"Hit@5: "
    f"{hit_at_5}/{total} "
    f"({100 * hit_at_5 / total:.2f}%)"
)

if ranks:

    reciprocal_ranks = [
        1.0 / rank
        for rank in ranks
    ]

    mrr = (
        sum(reciprocal_ranks)
        / len(TEST_CASES)
    )

else:

    mrr = 0.0

print(
    f"MRR: "
    f"{mrr:.3f}"
)

print("=" * 70)