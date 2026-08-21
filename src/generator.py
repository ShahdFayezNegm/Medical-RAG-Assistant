import os
import re
import time

import streamlit as st

from langchain_ollama import ChatOllama
from huggingface_hub import InferenceClient


class LlamaGenerator:
    """
    Medical RAG answer generator.

    Modes:

    LOCAL
        Ollama + llama3.1:8b

    DEPLOYMENT
        Hugging Face Inference Providers
        + meta-llama/Llama-3.1-8B-Instruct

    The RAG retrieval pipeline remains unchanged.
    """

    def __init__(
        self,
        model_name="llama3.1:8b",
        deployment_model="meta-llama/Llama-3.1-8B-Instruct",
        max_context_documents=2,
        max_context_chars=1200,
    ):

        self.local_model = model_name

        self.deployment_model = (
            deployment_model
        )

        self.max_context_documents = (
            max_context_documents
        )

        self.max_context_chars = (
            max_context_chars
        )

        # --------------------------------------------------
        # Detect deployment token
        # --------------------------------------------------

        self.hf_token = (
            os.getenv("HF_TOKEN")
            or self._get_streamlit_secret(
                "HF_TOKEN"
            )
        )

        self.use_huggingface = bool(
            self.hf_token
        )

        # --------------------------------------------------
        # Local Ollama
        # --------------------------------------------------

        self.ollama = None

        if not self.use_huggingface:

            self.ollama = ChatOllama(
                model=self.local_model,
                temperature=0.0,
                keep_alive="10m",
            )

        # --------------------------------------------------
        # Hugging Face
        # --------------------------------------------------

        self.hf_client = None

        if self.use_huggingface:

            self.hf_client = InferenceClient(
                api_key=self.hf_token,
                provider="auto",
            )

    # ======================================================
    # STREAMLIT SECRET
    # ======================================================

    @staticmethod
    def _get_streamlit_secret(
        key
    ):

        try:

            return st.secrets.get(
                key,
                None,
            )

        except Exception:

            return None

    # ======================================================
    # NORMALIZE
    # ======================================================

    @staticmethod
    def _normalize(
        text
    ):

        if not text:
            return ""

        text = str(text).lower()

        text = re.sub(
            r"[^a-z0-9\s\-]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    # ======================================================
    # COMPACT TEXT
    # ======================================================

    def _compact_text(
        self,
        text
    ):

        if not text:
            return ""

        text = " ".join(
            str(text).split()
        ).strip()

        if (
            len(text)
            <= self.max_context_chars
        ):
            return text

        shortened = text[
            :self.max_context_chars
        ]

        boundaries = [
            shortened.rfind(". "),
            shortened.rfind("; "),
            shortened.rfind(": "),
        ]

        last_boundary = max(
            boundaries
        )

        if (
            last_boundary
            >= int(
                self.max_context_chars
                * 0.60
            )
        ):

            shortened = shortened[
                :last_boundary + 1
            ]

        else:

            shortened = shortened.rstrip()

        return shortened + "..."

    # ======================================================
    # GENERAL TREATMENT QUESTION
    # ======================================================

    def _is_general_treatment_question(
        self,
        query
    ):

        q = query.lower().strip()

        hypertension = (
            "hypertension" in q
            or "high blood pressure" in q
            or "blood pressure" in q
        )

        treatment = any(
            phrase in q
            for phrase in [
                "recommended treatment",
                "recommended therapy",
                "treatment for",
                "treatment of",
                "management of hypertension",
                "management of high blood pressure",
                "how is hypertension treated",
                "how is high blood pressure treated",
            ]
        )

        explicit_step = any(
            step in q
            for step in [
                "step 1",
                "step 2",
                "step 3",
                "step 4",
            ]
        )

        return (
            hypertension
            and treatment
            and not explicit_step
        )

    # ======================================================
    # STANDARD CONTEXT
    # ======================================================

    def _build_context(
        self,
        evidence_documents
    ):

        parts = []

        for index, doc in enumerate(
            evidence_documents,
            1
        ):

            source = doc.metadata.get(
                "source",
                "Unknown",
            )

            page = doc.metadata.get(
                "page",
                "Unknown",
            )

            text = self._compact_text(
                doc.page_content
            )

            parts.append(
                f"""
Evidence {index}
Source: {source}
Page: {page}

{text}
"""
            )

        return "\n\n".join(
            parts
        )

    # ======================================================
    # GENERAL TREATMENT CONTEXT
    # ======================================================

    def _build_general_context(
        self,
        evidence_documents
    ):

        parts = []

        for index, doc in enumerate(
            evidence_documents,
            1
        ):

            source = doc.metadata.get(
                "source",
                "Unknown",
            )

            page = doc.metadata.get(
                "page",
                "Unknown",
            )

            text = self._compact_text(
                doc.page_content
            )

            parts.append(
                f"""
Evidence {index}
Source: {source}
Page: {page}

{text}
"""
            )

        return "\n\n".join(
            parts
        )

    # ======================================================
    # PROMPT
    # ======================================================

    def _build_prompt(
        self,
        query,
        context
    ):

        general_treatment = (
            self._is_general_treatment_question(
                query
            )
        )

        if general_treatment:

            instructions = """
The user asked a general question about hypertension treatment.

Important:
- Do not choose one treatment step arbitrarily.
- Preserve treatment conditions and subgroups.
- If the evidence contains multiple treatment stages,
  summarize the stages supported by the evidence.
- Do not invent missing treatment stages.
- Do not merge drugs from different stages into one recommendation.
"""

        else:

            instructions = """
Answer the specific question directly.
Use only the evidence supplied.
Do not add unsupported medical facts.
"""

        return f"""
You are a medical question-answering assistant.

{instructions}

STRICT RULES:

1. Use ONLY the provided evidence.
2. Do not use outside medical knowledge.
3. Do not guess.
4. Do not speculate.
5. Do not add unsupported facts.
6. Preserve conditions and qualifiers.
7. Do not call something "first-line" unless the evidence supports it.
8. Do not call a treatment "alone" unless the evidence says so.
9. Do not turn consequences into risk factors.
10. Keep the answer concise.
11. If the evidence is insufficient, say exactly:
"I don't have enough information."

Question:
{query}

Evidence:
{context}

Answer:
"""

    # ======================================================
    # CLEAN ANSWER
    # ======================================================

    @staticmethod
    def _clean_answer(
        answer
    ):

        if answer is None:

            return (
                "I don't have enough information."
            )

        answer = str(
            answer
        ).strip()

        answer = re.sub(
            r"\[SUPPORTING_DOCUMENTS:.*?\]",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        answer = re.sub(
            r"^\s*answer\s*:\s*",
            "",
            answer,
            flags=re.IGNORECASE,
        )

        return answer.strip()

    # ======================================================
    # LOCAL OLLAMA GENERATION
    # ======================================================

    def _generate_with_ollama(
        self,
        prompt
    ):

        response = self.ollama.invoke(
            prompt
        )

        return getattr(
            response,
            "content",
            response,
        )

    # ======================================================
    # HUGGING FACE GENERATION
    # ======================================================

    def _generate_with_huggingface(
        self,
        prompt
    ):

        response = (
            self.hf_client.chat.completions.create(
                model=self.deployment_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=0.0,
                max_tokens=500,
            )
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    # ======================================================
    # GENERATE
    # ======================================================

    def generate(
        self,
        query,
        documents
    ):

        if not documents:

            return {
                "answer":
                    "I don't have enough information.",
                "sources": [],
                "evidence": [],
            }

        # --------------------------------------------------
        # General questions get more evidence.
        # Normal questions get only top 2.
        # --------------------------------------------------

        if (
            self._is_general_treatment_question(
                query
            )
        ):

            evidence_documents = list(
                documents
            )[:5]

        else:

            evidence_documents = list(
                documents
            )[
                :self.max_context_documents
            ]

        if not evidence_documents:

            return {
                "answer":
                    "I don't have enough information.",
                "sources": [],
                "evidence": [],
            }

        context = self._build_context(
            evidence_documents
        )

        prompt = self._build_prompt(
            query,
            context
        )

        start_time = time.perf_counter()

        try:

            if self.use_huggingface:

                answer = (
                    self._generate_with_huggingface(
                        prompt
                    )
                )

            else:

                answer = (
                    self._generate_with_ollama(
                        prompt
                    )
                )

            answer = self._clean_answer(
                answer
            )

        except Exception as exc:

            answer = (
                f"Generation failed: {exc}"
            )

        generation_seconds = (
            time.perf_counter()
            - start_time
        )

        # --------------------------------------------------
        # Evidence
        # --------------------------------------------------

        evidence = []

        for index, doc in enumerate(
            evidence_documents,
            1
        ):

            evidence.append(
                {
                    "document":
                        index,

                    "source":
                        doc.metadata.get(
                            "source",
                            "Unknown",
                        ),

                    "page":
                        doc.metadata.get(
                            "page",
                            "Unknown",
                        ),

                    "text":
                        self._compact_text(
                            doc.page_content
                        ),

                    "rank":
                        doc.metadata.get(
                            "hybrid_rerank_rank",
                            index,
                        ),

                    "cross_encoder_score":
                        doc.metadata.get(
                            "cross_encoder_score",
                            None,
                        ),

                    "hybrid_rerank_score":
                        doc.metadata.get(
                            "hybrid_rerank_score",
                            None,
                        ),
                }
            )

        # --------------------------------------------------
        # Sources
        # --------------------------------------------------

        sources = []

        seen = set()

        for item in evidence:

            key = (
                item["source"],
                item["page"],
            )

            if key in seen:
                continue

            sources.append(
                {
                    "source":
                        item["source"],

                    "page":
                        item["page"],
                }
            )

            seen.add(
                key
            )

        return {
            "answer":
                answer,

            "sources":
                sources,

            "evidence":
                evidence,

            "generation_seconds":
                round(
                    generation_seconds,
                    2,
                ),

            "generator_mode":
                (
                    "huggingface"
                    if self.use_huggingface
                    else "ollama-local"
                ),
        }