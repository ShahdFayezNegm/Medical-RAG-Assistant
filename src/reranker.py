from __future__ import annotations

import re
from typing import List, Sequence

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Hybrid chunk-level reranker.

    Final ranking combines:

        CrossEncoder relevance
            +
        Query-term overlap
            +
        Medical entity overlap
            +
        Question-type phrase matching

    The CrossEncoder remains important, but it is no longer
    the only ranking signal.

    Public API is intentionally compatible with the existing
    HybridRetriever:

        self.reranker.rerank(
            query=query,
            documents=merged,
            top_k=k
        )
    """

    # ======================================================
    # INIT
    # ======================================================

    def __init__(
        self,
        model_name=(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        top_k=5,
        batch_size=16,
        max_length=512,
        unknown_threshold=-2.0,
    ):

        self.model_name = model_name
        self.top_k = top_k
        self.batch_size = batch_size
        self.max_length = max_length
        self.unknown_threshold = unknown_threshold

        # --------------------------------------------------
        # No deprecated automodel_args/tokenizer_args.
        # --------------------------------------------------

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
        )

        # --------------------------------------------------
        # Medical entities
        # --------------------------------------------------

        self.treatment_entities = [
            "metformin",
            "insulin",
            "sulfonylurea",
            "gliclazide",
            "glipizide",
            "pioglitazone",
            "sitagliptin",
            "dapagliflozin",
            "empagliflozin",
            "ace inhibitor",
            "arb",
            "ccb",
            "calcium channel blocker",
            "amlodipine",
            "lisinopril",
            "losartan",
            "thiazide",
            "thiazide-like diuretic",
            "diuretic",
        ]

        self.symptom_entities = [
            "polyuria",
            "polydipsia",
            "polyphagia",
            "weight loss",
            "fatigue",
            "blurred vision",
            "neuropathy",
            "retinopathy",
            "gastroparesis",
            "bloating",
            "vomiting",
            "sweating",
            "diarrhoea",
            "diarrhea",
            "hyperglycaemia",
            "hyperglycemia",
        ]

        self.risk_entities = [
            "smoking",
            "obesity",
            "dyslipidaemia",
            "dyslipidemia",
            "sedentary",
            "family history",
            "chronic kidney disease",
            "ckd",
            "diabetes",
            "stroke",
            "cardiovascular disease",
            "cardiovascular risk",
        ]

    # ======================================================
    # NORMALIZE
    # ======================================================

    @staticmethod
    def _normalize(text: str) -> str:

        if not text:
            return ""

        text = str(text).lower()

        # Preserve hyphenated words such as:
        # first-line
        # type-2
        # thiazide-like
        text = re.sub(
            r"[^a-z0-9\s\-]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ======================================================
    # QUESTION TYPE
    # ======================================================

    def _question_type(
        self,
        query: str
    ) -> str:

        q = self._normalize(
            query
        )

        if any(
            phrase in q
            for phrase in [
                "treatment",
                "therapy",
                "management",
                "medication",
                "drug",
                "first-line",
                "first line",
                "recommended treatment",
                "recommended therapy",
                "recommended",
            ]
        ):
            return "treatment"

        if any(
            phrase in q
            for phrase in [
                "symptom",
                "symptoms",
                "sign",
                "signs",
                "clinical features",
                "presentation",
            ]
        ):
            return "symptoms"

        if any(
            phrase in q
            for phrase in [
                "risk factor",
                "risk factors",
                "predispose",
                "predisposing",
                "at risk",
            ]
        ):
            return "risk_factors"

        if any(
            phrase in q
            for phrase in [
                "diagnosis",
                "diagnostic",
                "criteria",
                "test",
            ]
        ):
            return "diagnosis"

        return "general"

    # ======================================================
    # QUESTION TERMS
    # ======================================================

    def _question_terms(
        self,
        query: str
    ) -> set[str]:

        stopwords = {
            "what",
            "is",
            "are",
            "the",
            "for",
            "of",
            "a",
            "an",
            "in",
            "to",
            "and",
            "with",
            "on",
            "does",
            "do",
            "how",
            "which",
            "about",
            "from",
            "this",
            "that",
            "when",
            "who",
            "can",
            "could",
            "should",
            "please",
            "mentioned",
            "documents",
            "disease",
        }

        tokens = (
            self._normalize(
                query
            ).split()
        )

        return {
            token
            for token in tokens
            if token not in stopwords
            and len(token) > 2
        }

    # ======================================================
    # TOKEN OVERLAP
    # ======================================================

    def _query_overlap(
        self,
        query: str,
        text: str
    ) -> float:

        query_terms = self._question_terms(
            query
        )

        if not query_terms:
            return 0.0

        text_terms = set(
            self._normalize(
                text
            ).split()
        )

        overlap = (
            query_terms.intersection(
                text_terms
            )
        )

        return (
            len(overlap)
            / len(query_terms)
        )

    # ======================================================
    # MEDICAL ENTITY OVERLAP
    # ======================================================

    def _entity_score(
        self,
        text: str,
        question_type: str
    ) -> float:

        normalized = self._normalize(
            text
        )

        if question_type == "treatment":

            entities = (
                self.treatment_entities
            )

        elif question_type == "symptoms":

            entities = (
                self.symptom_entities
            )

        elif question_type == "risk_factors":

            entities = (
                self.risk_entities
            )

        else:

            entities = (
                self.treatment_entities
                + self.symptom_entities
                + self.risk_entities
            )

        matches = [
            entity
            for entity in entities
            if entity in normalized
        ]

        if not matches:
            return 0.0

        # Cap at 1.
        return min(
            len(matches) / 3.0,
            1.0
        )

    # ======================================================
    # EXACT / STRONG PHRASE SCORE
    # ======================================================

    def _phrase_score(
        self,
        query: str,
        text: str,
        question_type: str
    ) -> float:

        normalized = self._normalize(
            text
        )

        score = 0.0

        # --------------------------------------------------
        # Treatment
        # --------------------------------------------------

        if question_type == "treatment":

            # Very strong anchors.
            if (
                "first-line" in normalized
                or "first line" in normalized
            ):
                score += 1.00

            if (
                "initial therapy" in normalized
                or "initial treatment" in normalized
                or "initial management" in normalized
            ):
                score += 0.80

            if "step 1" in normalized:
                score += 0.50

            if "step 2" in normalized:
                score += 0.35

            if "step 3" in normalized:
                score += 0.35

            # Drug/treatment terminology.
            if "metformin" in normalized:
                score += 0.55

            if "ace inhibitor" in normalized:
                score += 0.30

            if " arb " in f" {normalized} ":
                score += 0.30

            if (
                "ccb" in normalized
                or "calcium channel blocker"
                in normalized
            ):
                score += 0.30

            if "thiazide-like" in normalized:
                score += 0.30

            if "recommended" in normalized:
                score += 0.15

            if "offer" in normalized:
                score += 0.10

        # --------------------------------------------------
        # Symptoms
        # --------------------------------------------------

        elif question_type == "symptoms":

            symptom_terms = [
                "symptom",
                "symptoms",
                "sign",
                "signs",
                "polyuria",
                "polydipsia",
                "polyphagia",
                "weight loss",
                "fatigue",
                "blurred vision",
                "neuropathy",
                "bloating",
                "vomiting",
                "sweating",
                "diarrhoea",
                "diarrhea",
                "hyperglycaemia",
                "hyperglycemia",
            ]

            matched = sum(
                1
                for term in symptom_terms
                if term in normalized
            )

            score += min(
                matched * 0.20,
                1.0
            )

        # --------------------------------------------------
        # Risk factors
        # --------------------------------------------------

        elif question_type == "risk_factors":

            risk_terms = [
                "risk factor",
                "risk factors",
                "modifiable",
                "associated with",
                "additional risk factors",
                "predisposing",
                "cardiovascular risk",
            ]

            matched = sum(
                1
                for term in risk_terms
                if term in normalized
            )

            score += min(
                matched * 0.25,
                1.0
            )

        return min(
            score,
            1.0
        )

    # ======================================================
    # DIRECT QUESTION MATCH SCORE
    # ======================================================

    def _direct_match_score(
        self,
        query: str,
        text: str
    ) -> float:

        q = self._normalize(
            query
        )

        t = self._normalize(
            text
        )

        score = 0.0

        # Exact phrase overlap.
        if q and q in t:
            score += 1.0

        # Important multi-word medical phrases.
        important_phrases = [
            "first-line treatment",
            "first line treatment",
            "initial treatment",
            "initial therapy",
            "type 2 diabetes",
            "high blood pressure",
            "recommended treatment",
            "risk factors",
            "symptoms of diabetes",
        ]

        for phrase in important_phrases:

            normalized_phrase = self._normalize(
                phrase
            )

            if normalized_phrase in q:
                if normalized_phrase in t:
                    score += 0.35

        return min(
            score,
            1.0
        )

    # ======================================================
    # NORMALIZE CROSSENCODER SCORES
    # ======================================================

    @staticmethod
    def _normalize_cross_scores(
        scores: list[float]
    ) -> list[float]:

        if not scores:
            return []

        minimum = min(
            scores
        )

        maximum = max(
            scores
        )

        if maximum == minimum:

            return [
                0.5
                for _ in scores
            ]

        return [
            (
                score - minimum
            )
            / (
                maximum - minimum
            )
            for score in scores
        ]

    # ======================================================
    # BUILD QUERY / CHUNK PAIRS
    # ======================================================

    def _build_pairs(
        self,
        query: str,
        documents: Sequence[Document]
    ) -> List[list[str]]:

        pairs = []

        for document in documents:

            chunk_text = (
                " ".join(
                    document.page_content.split()
                )
            )

            pairs.append(
                [
                    query,
                    chunk_text,
                ]
            )

        return pairs

    # ======================================================
    # RERANK
    # ======================================================

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: int | None = None
    ) -> List[Document]:
        """
        Hybrid reranking.

        Final score:

            0.45 CrossEncoder
          + 0.25 query-term overlap
          + 0.20 medical/entity relevance
          + 0.10 direct phrase match

        For highly specific treatment questions such as
        first-line treatment, strong phrase matches can also
        receive an additional bonus.

        This keeps CrossEncoder important without allowing a
        generic lexical mismatch to outrank the exact medical
        evidence.
        """

        if not query:
            return []

        if not documents:
            return []

        documents = list(
            documents
        )

        if top_k is None:
            top_k = self.top_k

        top_k = max(
            1,
            min(
                int(top_k),
                len(documents)
            )
        )

        # ==================================================
        # CrossEncoder
        # ==================================================

        pairs = self._build_pairs(
            query,
            documents
        )

        cross_scores_raw = (
            self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        )

        cross_scores = [
            float(score)
            for score in cross_scores_raw
        ]

        # Normalize within current candidate set.
        cross_normalized = (
            self._normalize_cross_scores(
                cross_scores
            )
        )

        question_type = (
            self._question_type(
                query
            )
        )

        scored = []

        # ==================================================
        # Hybrid scoring
        # ==================================================

        for (
            document,
            raw_cross,
            norm_cross,
        ) in zip(
            documents,
            cross_scores,
            cross_normalized,
        ):

            text = document.page_content

            overlap = (
                self._query_overlap(
                    query,
                    text
                )
            )

            entity_score = (
                self._entity_score(
                    text,
                    question_type
                )
            )

            phrase_score = (
                self._phrase_score(
                    query,
                    text,
                    question_type
                )
            )

            direct_score = (
                self._direct_match_score(
                    query,
                    text
                )
            )

            # ------------------------------------------------
            # Base hybrid score.
            # ------------------------------------------------

            final_score = (
                0.45 * norm_cross
                + 0.25 * overlap
                + 0.20 * entity_score
                + 0.10 * direct_score
            )

            # ------------------------------------------------
            # Strong first-line / initial-treatment bonus.
            #
            # This is deliberately question-specific.
            # ------------------------------------------------

            if question_type == "treatment":

                normalized_text = (
                    self._normalize(
                        text
                    )
                )

                normalized_query = (
                    self._normalize(
                        query
                    )
                )

                asks_first_line = (
                    "first-line" in normalized_query
                    or "first line" in normalized_query
                )

                if asks_first_line:

                    if (
                        "first-line" in normalized_text
                        or "first line" in normalized_text
                    ):

                        final_score += 0.35

                    if (
                        "initial therapy"
                        in normalized_text
                        or "initial treatment"
                        in normalized_text
                    ):

                        final_score += 0.25

                    if "metformin" in normalized_text:

                        final_score += 0.20

            # ------------------------------------------------
            # Store diagnostics.
            # ------------------------------------------------

            document.metadata = dict(
                document.metadata
            )

            document.metadata[
                "cross_encoder_score"
            ] = raw_cross

            document.metadata[
                "cross_encoder_normalized"
            ] = norm_cross

            document.metadata[
                "query_overlap_score"
            ] = overlap

            document.metadata[
                "entity_score"
            ] = entity_score

            document.metadata[
                "phrase_score"
            ] = phrase_score

            document.metadata[
                "direct_match_score"
            ] = direct_score

            document.metadata[
                "hybrid_rerank_score"
            ] = final_score

            scored.append(
                (
                    document,
                    final_score,
                    raw_cross,
                )
            )

        # ==================================================
        # Unknown rejection
        #
        # Use raw CrossEncoder signal for this.
        # ==================================================

        if not scored:

            return []

        best_raw_cross = max(
            item[2]
            for item in scored
        )

        if best_raw_cross < self.unknown_threshold:

            return []

        # ==================================================
        # Sort by hybrid score
        # ==================================================

        scored.sort(
            key=lambda item: item[1],
            reverse=True
        )

        selected = scored[
            :top_k
        ]

        results = []

        for rank, (
            document,
            hybrid_score,
            raw_cross,
        ) in enumerate(
            selected,
            start=1
        ):

            document.metadata = dict(
                document.metadata
            )

            document.metadata[
                "cross_encoder_rank"
            ] = rank

            document.metadata[
                "hybrid_rerank_rank"
            ] = rank

            document.metadata[
                "hybrid_rerank_score"
            ] = float(
                hybrid_score
            )

            document.metadata[
                "cross_encoder_score"
            ] = float(
                raw_cross
            )

            results.append(
                document
            )

        return results

    # ======================================================
    # DEBUG
    # ======================================================

    def print_results(
        self,
        documents: Sequence[Document]
    ) -> None:

        print(
            "\n"
            + "=" * 70
        )

        print(
            "HYBRID CHUNK-LEVEL RERANKING"
        )

        print(
            "=" * 70
        )

        for index, document in enumerate(
            documents,
            start=1
        ):

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                "Unknown"
            )

            raw_cross = document.metadata.get(
                "cross_encoder_score",
                0.0
            )

            hybrid_score = document.metadata.get(
                "hybrid_rerank_score",
                0.0
            )

            overlap = document.metadata.get(
                "query_overlap_score",
                0.0
            )

            entity_score = document.metadata.get(
                "entity_score",
                0.0
            )

            phrase_score = document.metadata.get(
                "phrase_score",
                0.0
            )

            chunk_text = " ".join(
                document.page_content.split()
            )

            if len(chunk_text) > 350:

                chunk_preview = (
                    chunk_text[:350]
                    + "..."
                )

            else:

                chunk_preview = chunk_text

            print(
                f"{index}. "
                f"{source} | "
                f"Page: {page}"
            )

            print(
                f"   "
                f"CrossEncoder: {raw_cross:.4f} | "
                f"Hybrid: {hybrid_score:.4f}"
            )

            print(
                f"   "
                f"Overlap: {overlap:.3f} | "
                f"Entity: {entity_score:.3f} | "
                f"Phrase: {phrase_score:.3f}"
            )

            print(
                f"   Chunk: {chunk_preview}"
            )

        print(
            "=" * 70
        )


# ==========================================================
# BACKWARD COMPATIBILITY
# ==========================================================

MedicalReranker = Reranker

