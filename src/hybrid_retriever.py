import re

from src.retriever import Retriever
from src.bm25_retriever import BM25Retriever
from src.reranker import Reranker


class HybridRetriever:
    """
    Hybrid Retrieval Pipeline

    Combines:
        1. FAISS semantic retrieval
        2. BM25 keyword retrieval
        3. Source filtering
        4. Chunk-level hybrid reranking
        5. Explicit treatment-step prioritization

    Important:
        General treatment routing is intentionally NOT used.

        Broad questions follow the normal retrieval pipeline,
        while explicit queries such as "step 2 treatment" can
        still receive step-aware prioritization.
    """

    def __init__(
        self,
        faiss_db,
        documents
    ):

        self.semantic = Retriever(
            faiss_db
        )

        self.keyword = BM25Retriever(
            documents
        )

        self.reranker = Reranker()

    # ==================================================
    # DETECT TARGET SOURCE
    # ==================================================

    def _detect_target_source(
        self,
        query
    ):

        q = query.lower()

        # --------------------------------------------------
        # Hypertension
        # --------------------------------------------------

        if (
            "hypertension" in q
            or "high blood pressure" in q
            or "blood pressure" in q
        ):
            return "hypertension_guideline.pdf"

        # --------------------------------------------------
        # Diabetes
        # --------------------------------------------------

        if (
            "type 2 diabetes" in q
            or "type-2 diabetes" in q
            or "diabetes" in q
            or "t2dm" in q
        ):
            return "diabetes_guideline.pdf"

        return None

    # ==================================================
    # DETECT EXPLICIT TREATMENT STEP
    # ==================================================

    def _detect_requested_step(
        self,
        query
    ):

        q = query.lower()

        match = re.search(
            r"\bstep\s*([1-4])\b",
            q
        )

        if match:
            return f"step {match.group(1)}"

        return None

    # ==================================================
    # DETECT TREATMENT STEP FROM DOCUMENT
    # ==================================================

    def _detect_treatment_step(
        self,
        document
    ):
        """
        Detect the treatment step represented by a chunk.

        Section headings and NICE recommendation numbers are
        preferred over incidental mentions of another step.
        """

        text = document.page_content.lower()

        metadata = document.metadata

        # --------------------------------------------------
        # Existing metadata
        # --------------------------------------------------

        existing_step = metadata.get(
            "treatment_step"
        )

        if existing_step:
            return existing_step

        # --------------------------------------------------
        # Normalize text
        # --------------------------------------------------

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        beginning = text[:500]

        # --------------------------------------------------
        # Strong section-heading detection
        # --------------------------------------------------

        for step_number in [4, 3, 2, 1]:

            pattern = (
                rf"\bstep\s*{step_number}"
                rf"\s*treatment\b"
            )

            if re.search(
                pattern,
                beginning,
                re.IGNORECASE
            ):

                return f"step {step_number}"

        # --------------------------------------------------
        # NICE recommendation numbers
        #
        # Step 1: 31-39
        # Step 2: 40-43
        # Step 3: 44-45
        # Step 4: 46
        # --------------------------------------------------

        recommendation_numbers = re.findall(
            r"\b1\.4\.(\d+)\b",
            text
        )

        recommendation_numbers = [
            int(number)
            for number in recommendation_numbers
        ]

        if recommendation_numbers:

            if any(
                46 <= number
                for number in recommendation_numbers
            ):
                return "step 4"

            if any(
                44 <= number <= 45
                for number in recommendation_numbers
            ):
                return "step 3"

            if any(
                40 <= number <= 43
                for number in recommendation_numbers
            ):
                return "step 2"

            if any(
                31 <= number <= 39
                for number in recommendation_numbers
            ):
                return "step 1"

        # --------------------------------------------------
        # Explicit heading anywhere in chunk
        # --------------------------------------------------

        matches = []

        for step_number in [1, 2, 3, 4]:

            pattern = (
                rf"\bstep\s*{step_number}"
                rf"\s*treatment\b"
            )

            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:

                matches.append(
                    (
                        match.start(),
                        step_number
                    )
                )

        if matches:

            matches.sort(
                key=lambda item: item[0]
            )

            return (
                f"step {matches[0][1]}"
            )

        return None

    # ==================================================
    # DOCUMENT KEY
    # ==================================================

    def _document_key(
        self,
        doc
    ):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        content = (
            doc.page_content.strip()
        )

        return (
            source,
            page,
            content
        )

    # ==================================================
    # SOURCE FILTER
    # ==================================================

    def _apply_source_filter(
        self,
        query,
        documents
    ):

        target_source = (
            self._detect_target_source(
                query
            )
        )

        if not target_source:

            return documents

        filtered = [
            doc
            for doc in documents
            if target_source.lower()
            in doc.metadata.get(
                "source",
                ""
            ).lower()
        ]

        if len(filtered) >= 5:

            print(
                f"Target source detected: "
                f"{target_source}"
            )

            print(
                f"Candidates after source filtering: "
                f"{len(filtered)}"
            )

            return filtered

        return documents

    # ==================================================
    # RETRIEVE
    # ==================================================

    def _retrieve_for_query(
        self,
        query,
        semantic_k=50,
        keyword_k=50,
        fetch_k=150
    ):

        # --------------------------------------------------
        # FAISS / MMR
        # --------------------------------------------------

        semantic_results = (
            self.semantic.mmr_search(
                query=query,
                k=semantic_k,
                fetch_k=fetch_k
            )
        )

        # --------------------------------------------------
        # BM25
        # --------------------------------------------------

        keyword_results = (
            self.keyword.search(
                query=query,
                k=keyword_k
            )
        )

        print(
            f"Semantic candidates: "
            f"{len(semantic_results)}"
        )

        print(
            f"Keyword candidates: "
            f"{len(keyword_results)}"
        )

        # --------------------------------------------------
        # Merge
        # --------------------------------------------------

        merged = []

        seen = set()

        for doc in (
            semantic_results
            + keyword_results
        ):

            key = self._document_key(
                doc
            )

            if key in seen:

                continue

            seen.add(
                key
            )

            merged.append(
                doc
            )

        print(
            f"Merged candidates: "
            f"{len(merged)}"
        )

        # --------------------------------------------------
        # Retrieval ranks
        # --------------------------------------------------

        semantic_rank = {}

        for rank, doc in enumerate(
            semantic_results,
            start=1
        ):

            semantic_rank[
                self._document_key(doc)
            ] = rank

        keyword_rank = {}

        for rank, doc in enumerate(
            keyword_results,
            start=1
        ):

            keyword_rank[
                self._document_key(doc)
            ] = rank

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------

        for doc in merged:

            key = self._document_key(
                doc
            )

            doc.metadata = dict(
                doc.metadata
            )

            doc.metadata[
                "faiss_rank"
            ] = semantic_rank.get(
                key
            )

            doc.metadata[
                "bm25_rank"
            ] = keyword_rank.get(
                key
            )

            doc.metadata[
                "treatment_step"
            ] = (
                self._detect_treatment_step(
                    doc
                )
            )

        return merged

    # ==================================================
    # SEARCH
    # ==================================================

    def search(
        self,
        query,
        k=5,
        semantic_k=50,
        keyword_k=50,
        fetch_k=150
    ):

        query = query.strip()

        if not query:

            return []

        # --------------------------------------------------
        # Hybrid retrieval
        # --------------------------------------------------

        merged = self._retrieve_for_query(
            query=query,
            semantic_k=semantic_k,
            keyword_k=keyword_k,
            fetch_k=fetch_k
        )

        # --------------------------------------------------
        # Source filtering
        # --------------------------------------------------

        merged = self._apply_source_filter(
            query,
            merged
        )

        # --------------------------------------------------
        # CrossEncoder + Hybrid Reranker
        # --------------------------------------------------

        reranked = self.reranker.rerank(
            query=query,
            documents=merged,
            top_k=min(
                max(k * 4, 20),
                len(merged)
            )
        )

        # --------------------------------------------------
        # Explicit step requested by user
        #
        # Example:
        # "What medications are offered in step 2?"
        # --------------------------------------------------

        requested_step = (
            self._detect_requested_step(
                query
            )
        )

        if requested_step:

            exact_step_docs = [
                doc
                for doc in reranked
                if doc.metadata.get(
                    "treatment_step"
                ) == requested_step
            ]

            other_docs = [
                doc
                for doc in reranked
                if doc.metadata.get(
                    "treatment_step"
                ) != requested_step
            ]

            reranked = (
                exact_step_docs
                + other_docs
            )

        # --------------------------------------------------
        # Final top K
        # --------------------------------------------------

        selected = reranked[:k]

        # --------------------------------------------------
        # Final rank metadata
        # --------------------------------------------------

        for rank, doc in enumerate(
            selected,
            start=1
        ):

            doc.metadata = dict(
                doc.metadata
            )

            doc.metadata[
                "hybrid_rerank_rank"
            ] = rank

        # ==================================================
        # DEBUG
        # ==================================================

        print(
            "\n"
            + "=" * 70
        )

        print(
            "FINAL HYBRID CHUNK-LEVEL RERANKING"
        )

        print(
            "=" * 70
        )

        for rank, doc in enumerate(
            selected,
            start=1
        ):

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            cross_score = doc.metadata.get(
                "cross_encoder_score",
                0.0
            )

            hybrid_score = doc.metadata.get(
                "hybrid_rerank_score",
                0.0
            )

            faiss_rank = doc.metadata.get(
                "faiss_rank",
                "N/A"
            )

            bm25_rank = doc.metadata.get(
                "bm25_rank",
                "N/A"
            )

            step = doc.metadata.get(
                "treatment_step",
                "N/A"
            )

            chunk_text = " ".join(
                doc.page_content.split()
            )

            if len(chunk_text) > 300:

                chunk_text = (
                    chunk_text[:300]
                    + "..."
                )

            print(
                f"{rank}. "
                f"{source} | "
                f"Page: {page} | "
                f"Step: {step}"
            )

            print(
                f"   CrossEncoder: "
                f"{cross_score:.4f} | "
                f"Hybrid: "
                f"{hybrid_score:.4f}"
            )

            print(
                f"   FAISS rank: "
                f"{faiss_rank} | "
                f"BM25 rank: "
                f"{bm25_rank}"
            )

            print(
                f"   Chunk: "
                f"{chunk_text}"
            )

        print(
            "=" * 70
        )

        return selected