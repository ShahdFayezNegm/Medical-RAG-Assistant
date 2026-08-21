import time
from pathlib import Path

from src.loader import PDFLoader
from src.splitter import DocumentSplitter
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.hybrid_retriever import HybridRetriever
from src.generator import LlamaGenerator


class MedicalRAGPipeline:
    """
    Medical RAG Pipeline

    Architecture:

        Existing PDFs
            |
            v
        PDF Loader
            |
            v
        Cleaning
            |
            v
        Chunking
            |
            v
        BGE Embeddings
            |
            v
        FAISS

        Question
            |
            +--> General hypertension treatment
            |       |
            |       +--> deterministic treatment pages
            |       |
            |       +--> Llama
            |
            +--> Normal question
                    |
                    +--> Hybrid Retrieval
                    |       FAISS + BM25
                    |       + CrossEncoder
                    |
                    +--> Llama

    Important:
    General hypertension treatment questions are handled
    separately because "step 1", "step 2", "step 3", and
    "step 4" contain very similar terminology.
    """

    def __init__(
        self,
        data_path="data",
        faiss_path="models/faiss_index",
    ):

        print("=" * 60)
        print("Building Medical RAG Pipeline...")
        print("=" * 60)

        self.data_path = data_path
        self.faiss_path = faiss_path

        # ==================================================
        # 1. Load documents
        # ==================================================

        loader = PDFLoader(data_path)

        documents = loader.load_documents()

        # ==================================================
        # 2. Split documents
        # ==================================================

        splitter = DocumentSplitter()

        chunks = splitter.split_documents(documents)

        # ==================================================
        # 3. Embeddings
        # ==================================================

        embedding_model = EmbeddingModel()

        self.embeddings = (
            embedding_model.get_embeddings()
        )

        # ==================================================
        # 4. Vector store
        # ==================================================

        self.vector_store = VectorStore(
            self.embeddings
        )

        # ==================================================
        # 5. Load/create FAISS
        # ==================================================

        faiss_dir = Path(self.faiss_path)

        index_file = faiss_dir / "index.faiss"
        metadata_file = faiss_dir / "index.pkl"

        faiss_exists = (
            faiss_dir.exists()
            and index_file.exists()
            and metadata_file.exists()
        )

        if faiss_exists:

            print()
            print("Loading existing FAISS index...")

            self.db = self.vector_store.load(
                self.faiss_path
            )

            print(
                "FAISS index loaded successfully."
            )

        else:

            print()
            print("FAISS index not found.")
            print("Creating a new FAISS index...")

            self.db = self.vector_store.create(
                chunks
            )

            self.vector_store.save(
                self.db,
                self.faiss_path
            )

            print(
                "FAISS index created and saved."
            )

        # ==================================================
        # 6. Current documents/chunks
        # ==================================================

        self.documents = documents
        self.chunks = chunks

        # ==================================================
        # 7. Hybrid retriever
        # ==================================================

        self.retriever = HybridRetriever(
            self.db,
            self.chunks
        )

        # ==================================================
        # 8. Llama
        # ==================================================

        self.generator = LlamaGenerator(
            model_name="llama3.1:8b"
        )

        print()
        print(
            f"Documents: {len(self.documents)}"
        )

        print(
            f"Chunks: {len(self.chunks)}"
        )

        print("=" * 60)

    # ======================================================
    # QUESTION ROUTING
    # ======================================================

    def _normalize_question(self, question):

        return (
            question
            .lower()
            .strip()
            .replace("?", "")
            .replace(".", "")
        )

    def _is_general_hypertension_treatment(
        self,
        question
    ):
        """
        Detect questions such as:

        What is the treatment for hypertension?
        What are the treatments for hypertension?
        How is hypertension treated?
        Treatment of hypertension?
        """

        q = self._normalize_question(
            question
        )

        hypertension_terms = [
            "hypertension",
            "high blood pressure",
        ]

        treatment_terms = [
            "treatment",
            "treat",
            "therapy",
            "management",
        ]

        has_hypertension = any(
            term in q
            for term in hypertension_terms
        )

        has_treatment = any(
            term in q
            for term in treatment_terms
        )

        # General treatment question only.
        # If the question explicitly asks for a step,
        # normal retrieval is used.
        explicit_step = any(
            term in q
            for term in [
                "step 1",
                "step 2",
                "step 3",
                "step 4",
            ]
        )

        return (
            has_hypertension
            and has_treatment
            and not explicit_step
        )

    # ======================================================
    # FIND HYPERTENSION GUIDELINE
    # ======================================================

    def _get_hypertension_chunks(self):
        """
        Get the actual chunks belonging to the
        hypertension guideline.

        We do NOT trust treatment_step metadata because
        the existing chunks have treatment_step=None.

        Instead we use:
            source filename
            page number
        """

        result = []

        for chunk in self.chunks:

            metadata = getattr(
                chunk,
                "metadata",
                {}
            ) or {}

            source = str(
                metadata.get(
                    "source",
                    ""
                )
            ).lower()

            page = metadata.get(
                "page"
            )

            if (
                "hypertension_guideline.pdf"
                in source
            ):

                try:
                    page_number = int(page)
                except (
                    TypeError,
                    ValueError
                ):
                    continue

                # The guideline's treatment section
                # is on these pages.
                if page_number in {
                    18,
                    19,
                    20,
                    21,
                }:

                    result.append(
                        chunk
                    )

        # Sort by page
        result.sort(
            key=lambda d: int(
                d.metadata.get(
                    "page",
                    0
                )
            )
        )

        return result

    # ======================================================
    # BUILD GENERAL HYPERTENSION EVIDENCE
    # ======================================================

    def _get_general_hypertension_evidence(
        self
    ):
        """
        Build deterministic evidence for:

            What is the treatment for hypertension?

        We deliberately use pages 18-21.

        Page 19:
            Step 1

        Page 20:
            Step 2

        Page 21:
            Step 3 + Step 4
        """

        chunks = (
            self._get_hypertension_chunks()
        )

        if not chunks:
            return []

        # --------------------------------------------------
        # Keep only the strongest relevant chunks.
        #
        # We prefer pages 19-21 because they contain
        # the treatment algorithm.
        # --------------------------------------------------

        preferred_pages = {
            19,
            20,
            21,
        }

        preferred = [
            chunk
            for chunk in chunks
            if int(
                chunk.metadata.get(
                    "page",
                    0
                )
            ) in preferred_pages
        ]

        if preferred:
            chunks = preferred

        # --------------------------------------------------
        # One chunk per page
        # --------------------------------------------------

        selected = []

        seen_pages = set()

        for chunk in chunks:

            try:
                page = int(
                    chunk.metadata.get(
                        "page"
                    )
                )
            except (
                TypeError,
                ValueError
            ):
                continue

            if page in seen_pages:
                continue

            seen_pages.add(page)

            selected.append(
                chunk
            )

        selected.sort(
            key=lambda d: int(
                d.metadata.get(
                    "page",
                    0
                )
            )
        )

        return selected

    # ======================================================
    # BUILD SOURCES
    # ======================================================

    def _build_sources(
        self,
        docs
    ):

        sources = []

        seen = set()

        for doc in docs:

            metadata = getattr(
                doc,
                "metadata",
                {}
            ) or {}

            source = metadata.get(
                "source",
                ""
            )

            page = metadata.get(
                "page",
                ""
            )

            key = (
                str(source),
                str(page)
            )

            if key in seen:
                continue

            seen.add(key)

            sources.append(
                {
                    "source": source,
                    "page": page,
                }
            )

        return sources

    # ======================================================
    # BUILD EVIDENCE
    # ======================================================

    def _build_evidence(
        self,
        docs
    ):

        evidence = []

        for index, doc in enumerate(
            docs,
            1
        ):

            metadata = getattr(
                doc,
                "metadata",
                {}
            ) or {}

            evidence.append(
                {
                    "document": index,
                    "source": metadata.get(
                        "source",
                        ""
                    ),
                    "page": metadata.get(
                        "page",
                        ""
                    ),
                }
            )

        return evidence

    # ======================================================
    # GENERAL HYPERTENSION ANSWER
    # ======================================================

    def _answer_general_hypertension(
        self,
        question
    ):

        print()
        print("=" * 60)
        print(
            "GENERAL HYPERTENSION TREATMENT ROUTING"
        )
        print("=" * 60)

        docs = (
            self._get_general_hypertension_evidence()
        )

        if not docs:

            return {
                "answer":
                    "No hypertension treatment evidence was found.",
                "sources": [],
                "evidence": [],
            }

        print()
        print(
            "Selected hypertension treatment evidence:"
        )

        for doc in docs:

            metadata = getattr(
                doc,
                "metadata",
                {}
            ) or {}

            print(
                f"- Page "
                f"{metadata.get('page')} "
                f"| "
                f"{metadata.get('source')}"
            )

        # ==================================================
        # IMPORTANT:
        #
        # We construct a dedicated context.
        # ==================================================

        context_parts = []

        for doc in docs:

            metadata = getattr(
                doc,
                "metadata",
                {}
            ) or {}

            page = metadata.get(
                "page",
                ""
            )

            source = metadata.get(
                "source",
                ""
            )

            text = (
                getattr(
                    doc,
                    "page_content",
                    ""
                )
                or ""
            )

            context_parts.append(
                f"""
SOURCE: {source}
PAGE: {page}

TEXT:
{text}
"""
            )

        context = "\n".join(
            context_parts
        )

        # ==================================================
        # Dedicated prompt
        # ==================================================

        prompt = f"""
You are a medical guideline question-answering assistant.

Answer ONLY from the provided evidence.

USER QUESTION:
{question}

IMPORTANT INSTRUCTION:

The user is asking for the GENERAL treatment algorithm
for hypertension.

Do NOT answer only with what happens after treatment
fails.

You MUST organize the answer by treatment steps:

Step 1
Step 2
Step 3
Step 4

For each step, state the treatment described by the
guideline.

Do not invent recommendations.

Do not use medical knowledge that is not present in
the evidence.

Do not confuse:

"Step 1 treatment"

with

"what to do if Step 1 fails".

If the evidence does not contain enough information
for a particular step, explicitly say:

"Not enough information in the provided evidence."

Keep the answer concise and clinically precise.

EVIDENCE:
{context}

ANSWER:
"""

        # ==================================================
        # Generate
        # ==================================================

        generation_start = time.perf_counter()

        result = self.generator.generate(
            question,
            docs,
            custom_prompt=prompt
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        # ==================================================
        # Normalize result
        # ==================================================

        if isinstance(
            result,
            dict
        ):

            answer = result.get(
                "answer",
                ""
            )

        else:

            answer = str(
                result
            )

        sources = self._build_sources(
            docs
        )

        evidence = self._build_evidence(
            docs
        )

        return {
            "answer": answer.strip(),
            "sources": sources,
            "evidence": evidence,
            "timing": {
                "retrieval_seconds": 0.0,
                "generation_seconds":
                    round(
                        generation_time,
                        2
                    ),
                "total_seconds":
                    round(
                        generation_time,
                        2
                    ),
            },
        }

    # ======================================================
    # ADD PDF
    # ======================================================

    def add_pdf(
        self,
        pdf_path
    ):

        pdf_path = Path(
            pdf_path
        )

        if not pdf_path.exists():

            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if not pdf_path.is_file():

            raise ValueError(
                f"Path is not a file: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":

            raise ValueError(
                "Only PDF files are supported."
            )

        print()
        print("=" * 60)
        print("ADDING NEW PDF")
        print("=" * 60)

        print(
            f"PDF: {pdf_path.name}"
        )

        # ==================================================
        # Load ONLY new PDF
        # ==================================================

        loader = PDFLoader()

        new_documents = loader.load_pdf(
            pdf_path
        )

        if not new_documents:

            raise ValueError(
                "No readable pages were found "
                "in the uploaded PDF."
            )

        for doc in new_documents:

            doc.metadata = dict(
                doc.metadata
            )

            doc.metadata[
                "source"
            ] = pdf_path.name

        # ==================================================
        # Split
        # ==================================================

        splitter = DocumentSplitter()

        new_chunks = splitter.split_documents(
            new_documents
        )

        if not new_chunks:

            raise ValueError(
                "No chunks were created from "
                "the uploaded PDF."
            )

        # ==================================================
        # Add to FAISS
        # ==================================================

        self.db.add_documents(
            new_chunks
        )

        # ==================================================
        # Save
        # ==================================================

        self.vector_store.save(
            self.db,
            self.faiss_path
        )

        # ==================================================
        # Update memory
        # ==================================================

        self.documents.extend(
            new_documents
        )

        self.chunks.extend(
            new_chunks
        )

        # ==================================================
        # Rebuild retriever
        # ==================================================

        self.retriever = HybridRetriever(
            self.db,
            self.chunks
        )

        print()
        print(
            f"Added pages: "
            f"{len(new_documents)}"
        )

        print(
            f"Added chunks: "
            f"{len(new_chunks)}"
        )

        print(
            f"Total documents: "
            f"{len(self.documents)}"
        )

        print(
            f"Total chunks: "
            f"{len(self.chunks)}"
        )

        print(
            "FAISS index updated successfully."
        )

        print("=" * 60)

        return {
            "filename": pdf_path.name,
            "pages_added":
                len(new_documents),
            "chunks_added":
                len(new_chunks),
            "total_documents":
                len(self.documents),
            "total_chunks":
                len(self.chunks),
        }

    # ======================================================
    # ASK
    # ======================================================

    def ask(
        self,
        question
    ):

        if (
            not question
            or not question.strip()
        ):

            return {
                "answer":
                    "Please enter a question.",
                "sources": [],
                "evidence": [],
                "timing": {
                    "retrieval_seconds": 0.0,
                    "generation_seconds": 0.0,
                    "total_seconds": 0.0,
                },
            }

        # ==================================================
        # SPECIAL ROUTE
        # ==================================================

        if self._is_general_hypertension_treatment(
            question
        ):

            return (
                self._answer_general_hypertension(
                    question
                )
            )

        # ==================================================
        # NORMAL RAG
        # ==================================================

        retrieval_start = time.perf_counter()

        docs = self.retriever.search(
            question,
            k=5
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        # ==================================================
        # Generation
        # ==================================================

        generation_start = time.perf_counter()

        result = self.generator.generate(
            question,
            docs
        )

        generation_time = (
            time.perf_counter()
            - generation_start
        )

        total_time = (
            retrieval_time
            + generation_time
        )

        # ==================================================
        # Result
        # ==================================================

        if not isinstance(
            result,
            dict
        ):

            result = {
                "answer": str(
                    result
                )
            }

        result["sources"] = (
            result.get(
                "sources"
            )
            or self._build_sources(
                docs
            )
        )

        result["evidence"] = (
            result.get(
                "evidence"
            )
            or self._build_evidence(
                docs
            )
        )

        result["timing"] = {
            "retrieval_seconds":
                round(
                    retrieval_time,
                    2
                ),

            "generation_seconds":
                round(
                    generation_time,
                    2
                ),

            "total_seconds":
                round(
                    total_time,
                    2
                ),
        }

        print()
        print("=" * 60)
        print("RAG TIMING")
        print("=" * 60)

        print(
            f"Retrieval time: "
            f"{retrieval_time:.2f} seconds"
        )

        print(
            f"Generation time: "
            f"{generation_time:.2f} seconds"
        )

        print(
            f"Total time: "
            f"{total_time:.2f} seconds"
        )

        print("=" * 60)

        return result