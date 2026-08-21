from pathlib import Path

import streamlit as st

from src.rag_pipeline import MedicalRAGPipeline


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_question" not in st.session_state:
    st.session_state.last_question = ""

if "upload_message" not in st.session_state:
    st.session_state.upload_message = None


# ==========================================================
# LOAD PIPELINE
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_pipeline():

    return MedicalRAGPipeline(
        data_path="data",
        faiss_path="models/faiss_index",
    )


try:

    if st.session_state.pipeline is None:

        with st.spinner(
            "Loading Medical RAG Assistant..."
        ):
            st.session_state.pipeline = load_pipeline()

    pipeline = st.session_state.pipeline

except Exception as exc:

    st.error("Failed to load the Medical RAG pipeline.")
    st.exception(exc)
    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    st.caption(
        "Upload a medical PDF to add it to the existing knowledge base."
    )

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Maximum file size: 200 MB.",
    )

    if uploaded_file is not None:

        st.info(
            f"Selected: {uploaded_file.name}"
        )

    add_pdf_clicked = st.button(
        "➕ Add to Knowledge Base",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None,
    )

    if add_pdf_clicked and uploaded_file:

        data_dir = Path("data")
        data_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        pdf_path = data_dir / uploaded_file.name

        try:

            with open(
                pdf_path,
                "wb",
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )

            with st.spinner(
                "Processing PDF and updating the knowledge base..."
            ):

                upload_result = pipeline.add_pdf(
                    pdf_path
                )

            st.session_state.upload_message = upload_result

            st.success(
                "PDF added successfully."
            )

            st.rerun()

        except Exception as exc:

            st.error(
                "Failed to add the PDF."
            )

            st.exception(exc)

    st.divider()

    # ======================================================
    # KNOWLEDGE BASE STATS
    # ======================================================

    st.header("📊 Knowledge Base Stats")

    try:
        document_count = len(pipeline.documents)
    except Exception:
        document_count = "—"

    try:
        chunk_count = len(pipeline.chunks)
    except Exception:
        chunk_count = "—"

    stat1, stat2 = st.columns(2)

    with stat1:
        st.metric(
            "Documents",
            document_count,
        )

    with stat2:
        st.metric(
            "Chunks",
            chunk_count,
        )

    st.divider()

    # ======================================================
    # PIPELINE
    # ======================================================

    st.header("Pipeline")

    st.success("FAISS")
    st.success("BM25")
    st.success("CrossEncoder")
    st.success("Llama 3.1")

    st.caption(
        "Evidence-grounded medical QA"
    )


# ==========================================================
# UPLOAD RESULT
# ==========================================================

if st.session_state.upload_message:

    info = st.session_state.upload_message

    st.success(
        f"{info.get('filename', 'PDF')} added successfully."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Pages Added",
            info.get("pages_added", 0),
        )

    with c2:
        st.metric(
            "Chunks Added",
            info.get("chunks_added", 0),
        )

    with c3:
        st.metric(
            "Total Chunks",
            info.get(
                "total_chunks",
                len(pipeline.chunks),
            ),
        )

    st.session_state.upload_message = None


# ==========================================================
# HERO
# ==========================================================

st.title(
    "🩺 Medical RAG Assistant"
)

st.caption(
    "Evidence-grounded medical question answering "
    "using FAISS, BM25, CrossEncoder reranking, "
    "and Llama 3.1."
)


# ==========================================================
# QUESTION
# ==========================================================

st.subheader(
    "🔎 Ask a Medical Question"
)

st.caption(
    "Ask a question about information contained "
    "in the indexed medical documents."
)

question = st.text_area(
    "Question",
    value=st.session_state.last_question,
    placeholder=(
        "Example: What is the first-line treatment "
        "for type 2 diabetes?"
    ),
    height=110,
)


ask_col, clear_col, _ = st.columns(
    [1, 1, 5]
)

with ask_col:

    ask_clicked = st.button(
        "🔍 Ask",
        type="primary",
        use_container_width=True,
    )

with clear_col:

    clear_clicked = st.button(
        "Clear",
        use_container_width=True,
    )


# ==========================================================
# CLEAR
# ==========================================================

if clear_clicked:

    st.session_state.last_question = ""
    st.session_state.last_result = None

    st.rerun()


# ==========================================================
# ASK QUESTION
# ==========================================================

if ask_clicked:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a medical question."
        )

    else:

        st.session_state.last_question = question

        with st.spinner(
            "Searching the knowledge base and generating the answer..."
        ):

            try:

                result = pipeline.ask(
                    question
                )

                st.session_state.last_result = result

            except Exception as exc:

                st.error(
                    "Failed to generate the answer."
                )

                st.exception(exc)


# ==========================================================
# RESULT
# ==========================================================

result = st.session_state.last_result


if result is not None:

    st.divider()

    # ======================================================
    # ANSWER
    # ======================================================

    st.subheader(
        "💬 Answer"
    )

    with st.container(border=True):

        st.caption(
            "Grounded Answer"
        )

        st.write(
            str(
                result.get(
                    "answer",
                    "I don't have enough information.",
                )
            )
        )

    # ======================================================
    # PERFORMANCE
    # ======================================================

    timing = result.get(
        "timing",
        {}
    )

    retrieval_time = timing.get(
        "retrieval_seconds",
        0.0,
    )

    generation_time = timing.get(
        "generation_seconds",
        0.0,
    )

    total_time = timing.get(
        "total_seconds",
        0.0,
    )

    st.subheader(
        "⚡ Performance"
    )

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "Retrieval",
            f"{float(retrieval_time):.2f}s",
        )

    with p2:
        st.metric(
            "Generation",
            f"{float(generation_time):.2f}s",
        )

    with p3:
        st.metric(
            "Total",
            f"{float(total_time):.2f}s",
        )

    # ======================================================
    # SUPPORTING EVIDENCE
    # ======================================================

    evidence = result.get(
        "evidence",
        []
    )

    if evidence:

        st.subheader(
            "📖 Supporting Evidence"
        )

        for index, item in enumerate(
            evidence,
            1,
        ):

            source = str(
                item.get(
                    "source",
                    "Unknown",
                )
            )

            page = str(
                item.get(
                    "page",
                    "Unknown",
                )
            )

            evidence_text = str(
                item.get(
                    "text",
                    "",
                )
            ).strip()

            with st.container(border=True):

                st.markdown(
                    f"**Evidence {index} • Page {page}**"
                )

                st.caption(
                    f"Source: {source} | Page: {page}"
                )

                # Never place PDF text inside HTML.
                st.write(
                    evidence_text
                )

    # ======================================================
    # SOURCES
    # ======================================================

    sources = result.get(
        "sources",
        []
    )

    if sources:

        st.subheader(
            "📚 Sources"
        )

        for index, source in enumerate(
            sources,
            1,
        ):

            source_name = str(
                source.get(
                    "source",
                    "Unknown",
                )
            )

            source_page = str(
                source.get(
                    "page",
                    "Unknown",
                )
            )

            with st.container(border=True):

                st.markdown(
                    f"**{index}. {source_name}**"
                )

                st.caption(
                    f"Page {source_page}"
                )

    # ======================================================
    # RAG DETAILS
    # ======================================================

    with st.expander(
        "🔬 RAG Details"
    ):

        st.write(
            "FAISS provides semantic retrieval."
        )

        st.write(
            "BM25 provides keyword retrieval."
        )

        st.write(
            "The CrossEncoder reranks retrieved chunks."
        )

        st.write(
            "Llama 3.1 generates the grounded answer."
        )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Medical RAG Assistant • FAISS • BM25 • "
    "CrossEncoder • Llama 3.1"
)

st.caption(
    "Answers are generated only from the indexed medical documents."
)