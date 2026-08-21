# 🩺 Medical RAG Assistant

> Evidence-grounded medical question answering using **FAISS + BM25 + CrossEncoder reranking + Llama 3.1**.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/BM25-Keyword%20Retrieval-orange.svg)](https://en.wikipedia.org/wiki/Okapi_BM25)
[![Llama](https://img.shields.io/badge/Llama%203.1-8B-purple.svg)](https://ollama.com/)

---

## 📌 Overview

Medical RAG Assistant is a Retrieval-Augmented Generation system for answering questions from an indexed collection of medical PDF documents.

Instead of asking the language model to answer directly from its internal knowledge, the system first retrieves relevant evidence from the medical knowledge base and then generates an answer grounded in the retrieved passages.

The application also supports adding new medical PDFs to the existing knowledge base and making their content searchable.

---

## ✨ Key Features

- 📚 **Medical PDF Knowledge Base** — index and search a growing collection of medical documents.
- 🔎 **Hybrid Retrieval** — combines semantic FAISS retrieval with BM25 keyword retrieval.
- 🎯 **CrossEncoder Reranking** — reranks retrieved chunks at chunk level.
- 🧠 **Hybrid Relevance Scoring** — combines multiple relevance signals before final selection.
- 🛡️ **Grounded Answers** — instructs the LLM to use only the retrieved evidence.
- 📖 **Evidence & Sources** — displays supporting documents and page numbers.
- ➕ **Incremental PDF Ingestion** — add new medical PDFs from the Streamlit interface.
- ⏱️ **Performance Tracking** — displays retrieval, generation, and total response time.
- 🚫 **Unknown Question Rejection** — unsupported questions can be rejected instead of forcing an answer.
- 💻 **Local LLM Support** — Ollama + Llama 3.1 8B.
- ☁️ **Hosted LLM Support** — Hugging Face Inference Providers for deployment.
- 🧪 **Comprehensive Testing** — includes loader, splitter, vector-store, retrieval, BM25, PDF ingestion, RAG, and evaluation tests.

---

# 🏗️ Architecture

```text
                         ┌──────────────────────┐
                         │     Medical PDFs     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     PDF Loader       │
                         │ Cleaning + Metadata  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Document Splitter   │
                         │      913 chunks      │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌──────────────────┐             ┌──────────────────┐
          │      FAISS       │             │       BM25       │
          │ Semantic Search  │             │ Keyword Search   │
          └────────┬─────────┘             └────────┬─────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   ▼
                         ┌──────────────────────┐
                         │   Hybrid Retrieval   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    CrossEncoder      │
                         │    Chunk Reranking   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Relevance + Filtering│
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Llama 3.1 8B     │
                         │ Evidence-grounded    │
                         │     Generation        │
                         └──────────┬───────────┘
                                    │
                                    ▼
             ┌─────────────────────────────────────────┐
             │ Answer + Evidence + Sources + Timing   │
             └─────────────────────────────────────────┘
```

---

# 📁 Project Structure

```text
Medical-RAG-Assistant/
│
├── .gitignore
├── app.py
├── build_faiss.py
├── README.md
├── requirements.txt
│
├── .streamlit/
│   └── secrets.toml        # Local only — not committed to Git
│
├── data/
│   └── *.pdf
│
├── models/
│   └── faiss_index/
│       ├── index.faiss
│       └── index.pkl
│
├── src/
│   ├── bm25_retriever.py
│   ├── cleaner.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── hybrid_retriever.py
│   ├── loader.py
│   ├── rag_pipeline.py
│   ├── reranker.py
│   ├── retriever.py
│   ├── splitter.py
│   └── vector_store.py
│
└── tests/
    ├── test_add_pdf.py
    ├── test_bm25.py
    ├── test_evaluation.py
    ├── test_loader.py
    ├── test_mmrr.py
    ├── test_rag_multiple.py
    ├── test_rag.py
    ├── test_retrievers.py
    ├── test_splitter.py
    └── test_vector_store.py
```

> **Security note:** `.streamlit/secrets.toml` is required locally for secrets such as `HF_TOKEN`, but it should **never be committed to GitHub**.

---

# 🔧 Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.12 |
| User Interface | Streamlit |
| Semantic Retrieval | FAISS |
| Keyword Retrieval | BM25 |
| Reranking | Sentence-Transformers CrossEncoder |
| Embeddings | Sentence-Transformers embedding model |
| LLM | Llama 3.1 8B |
| Local Inference | Ollama |
| Hosted Inference | Hugging Face Inference Providers |
| PDF Processing | LangChain Community PDF Loaders |
| Vector Store | FAISS |
| Evaluation | Hit@1, Hit@3, Hit@5, MRR, Unknown Rejection |

---

# 🔎 Retrieval Pipeline

## 1. FAISS Semantic Retrieval

FAISS is used for semantic retrieval based on vector similarity.

This allows the system to retrieve relevant passages even when the wording of the user's question differs from the wording used in the source document.

---

## 2. BM25 Keyword Retrieval

BM25 provides keyword-based retrieval.

This is especially useful for:

- Medical terminology
- Drug names
- Clinical terms
- Exact phrases
- Guideline terminology
- Recommendation numbers

---

## 3. Hybrid Retrieval

The system combines semantic FAISS results and BM25 keyword results.

The results are merged and deduplicated before reranking.

```text
Semantic Search
       +
Keyword Search
       =
Hybrid Retrieval
```

---

## 4. CrossEncoder Reranking

A CrossEncoder reranks the merged candidates based on the relevance between the user question and each retrieved chunk.

This improves the quality of the final evidence passed to the generator.

---

## 5. Relevance Filtering

After reranking, the system applies retrieval and relevance signals to select the most useful evidence chunks.

The system also supports source-aware filtering for known medical domains such as diabetes and hypertension.

---

## 6. Grounded Generation

Only the selected evidence is passed to Llama 3.1.

The generator is explicitly instructed to:

```text
Use ONLY the provided evidence.
Do not use outside medical knowledge.
Do not guess.
Do not speculate.
Do not add unsupported medical facts.
Preserve conditions and qualifiers.
```

If the evidence is insufficient, the system can return:

```text
I don't have enough information.
```

---

# 📚 Knowledge Base

The application supports dynamic medical PDF ingestion.

The workflow is:

```text
Upload PDF
     ↓
Document Loading
     ↓
Cleaning
     ↓
Chunking
     ↓
Embeddings
     ↓
FAISS Index
     ↓
BM25 Index
     ↓
Knowledge Base Updated
```

During development, the knowledge base increased from:

```text
289 Documents
867 Chunks
```

to:

```text
301 Documents
913 Chunks
```

after adding an additional NHS heart-attack document.

The newly uploaded document was successfully retrieved and used to generate a grounded answer.

---

# ➕ Incremental PDF Ingestion

A new medical PDF can be added directly from the Streamlit sidebar.

Workflow:

```text
Select PDF
     ↓
Add to Knowledge Base
     ↓
Load + Clean
     ↓
Split into Chunks
     ↓
Update FAISS
     ↓
Refresh BM25
     ↓
Document Searchable
```

This allows the knowledge base to grow without rebuilding the entire application manually from scratch.

---

# ✅ Example

## Question

```text
What is the first-line treatment for type 2 diabetes?
```

## Grounded Answer

```text
The first-line treatment for type 2 diabetes is metformin.
```

## Supporting Evidence

```text
Source: data\diabetes_guideline.pdf
Page: 68
```

The application displays the retrieved evidence together with the generated response.

---

# ❤️ New Medical Document Test

An additional NHS heart-attack document was uploaded through the application.

Question:

```text
What are the symptoms of a heart attack?
```

The system successfully retrieved evidence from:

```text
2025.04.23_NHS_HUHY_employer-toolkit.pdf
Page 2
```

This demonstrates the complete workflow:

```text
New PDF
   ↓
Knowledge Base Update
   ↓
FAISS + BM25 Retrieval
   ↓
CrossEncoder Reranking
   ↓
Grounded Llama Generation
   ↓
Evidence + Sources
```

---

# 📊 Evaluation

The current evaluation suite achieved:

| Metric | Result |
|---|---:|
| Hit@1 | **75.00%** |
| Hit@3 | **100.00%** |
| Hit@5 | **100.00%** |
| MRR | **0.875** |
| Unknown Rejection | **100.00%** |

These metrics are based on the project's current evaluation set and represent a project benchmark rather than a clinical validation study.

---

# 📈 Evaluation Metrics

## Hit@1

Percentage of evaluation questions where the expected source page is ranked first.

## Hit@3

Percentage of evaluation questions where the expected source page appears in the top three.

## Hit@5

Percentage of evaluation questions where the expected source page appears in the top five.

## MRR

Mean Reciprocal Rank measures how early the correct evidence appears in the retrieval ranking.

## Unknown Rejection

Measures whether unsupported questions are rejected instead of receiving an unsupported answer.

---

# 🛡️ Grounding Strategy

The generator is explicitly instructed to:

- Use only the provided evidence
- Avoid outside medical knowledge
- Avoid unsupported inference
- Preserve treatment conditions and qualifiers
- Avoid inventing sources or page numbers
- Reject questions when the evidence is insufficient

Unsupported questions return:

```text
I don't have enough information.
```

This grounding strategy is designed to reduce unsupported generation.

---

# ⚡ Performance

The application tracks:

- Retrieval time
- Generation time
- Total response time

Example hosted inference performance observed during testing:

```text
Retrieval:   2–10 seconds
Generation:  ~2 seconds
Total:       ~10 seconds
```

Local Ollama generation can be significantly slower depending on hardware.

---

# 🤖 LLM Modes

## Local Mode

```text
User
 ↓
RAG Pipeline
 ↓
Ollama
 ↓
Llama 3.1 8B
```

Used for local development and testing.

---

## Hosted Mode

```text
User
 ↓
Streamlit
 ↓
RAG Pipeline
 ↓
Hugging Face Inference Providers
 ↓
Llama 3.1 8B Instruct
```

The application automatically uses hosted inference when `HF_TOKEN` is configured.

---

# 🖥️ Streamlit Interface

The application provides:

- Medical PDF upload
- Knowledge Base statistics
- Medical question input
- Grounded answers
- Retrieval timing
- Generation timing
- Supporting evidence
- Source documents
- Page numbers

---

# 🚀 Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Medical-RAG-Assistant.git
cd Medical-RAG-Assistant
```

## 2. Create the Virtual Environment

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

## 4. Install and Run Ollama

The local development version uses:

```text
Llama 3.1 8B
```

Pull the model:

```powershell
ollama pull llama3.1:8b
```

Run Ollama:

```powershell
ollama serve
```

In another terminal:

```powershell
ollama list
```

## 5. Build the FAISS Index

If the FAISS index needs to be rebuilt:

```powershell
python build_faiss.py
```

## 6. Run the Evaluation

```powershell
python tests/test_evaluation.py
```

Expected benchmark for the current tested version:

```text
Hit@1: 3/4 (75.00%)
Hit@3: 4/4 (100.00%)
Hit@5: 4/4 (100.00%)
MRR: 0.875
Unknown Rejection: 1/1 (100.00%)
```

## 7. Run the Streamlit Application

```powershell
python -m streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# ☁️ Public Deployment

The public deployment architecture is:

```text
GitHub Repository
        ↓
Streamlit Community Cloud
        ↓
Hugging Face Inference Providers
        ↓
Llama 3.1 8B Instruct
```

Configure the Streamlit secret:

```toml
HF_TOKEN = "hf_your_token_here"
```

Never commit the real token to GitHub.

### Deployment Checklist

```text
[ ] Push project to GitHub
[ ] Add requirements.txt
[ ] Add .gitignore
[ ] Exclude secrets
[ ] Exclude local model weights
[ ] Prepare FAISS index / knowledge-base files
[ ] Configure HF_TOKEN in Streamlit Secrets
[ ] Deploy app.py
[ ] Test PDF upload
[ ] Test retrieval
[ ] Test grounded generation
[ ] Verify sources and evidence
[ ] Share public streamlit.app URL
```

---

# 🔒 Security & Privacy

Do not commit:

```text
.venv/
__pycache__/
.env
.streamlit/secrets.toml
API keys
Ollama credentials
Private medical documents
Private patient information
Private model weights
```

For a public medical-document demo, only use documents you are permitted to redistribute.

Do not upload personally identifiable patient information.

---

# 🧪 Testing

The project contains a comprehensive test suite covering document ingestion, preprocessing, retrieval, indexing, and end-to-end RAG behavior.

## Test Suite

| Test | Purpose |
|---|---|
| `test_loader.py` | Tests PDF loading and document cleaning |
| `test_splitter.py` | Tests document chunking and splitting |
| `test_vector_store.py` | Tests vector-store creation and loading |
| `test_bm25.py` | Tests BM25 keyword retrieval |
| `test_retrievers.py` | Tests retrieval components |
| `test_mmrr.py` | Tests MMR-based semantic retrieval |
| `test_add_pdf.py` | Tests adding a new PDF to the knowledge base |
| `test_rag.py` | Tests the RAG pipeline |
| `test_rag_multiple.py` | Tests RAG behavior across multiple questions |
| `test_evaluation.py` | Evaluates retrieval ranking and unknown-question rejection |

---

## Run Individual Tests

```powershell
python tests/test_loader.py
python tests/test_splitter.py
python tests/test_vector_store.py
python tests/test_bm25.py
python tests/test_retrievers.py
python tests/test_mmrr.py
python tests/test_add_pdf.py
python tests/test_rag.py
python tests/test_rag_multiple.py
python tests/test_evaluation.py
```

---

## Run All Tests with Pytest

```powershell
python -m pytest tests/
```

---

# 🔬 Example Questions

```text
What is the first-line treatment for type 2 diabetes?
```

```text
What are the symptoms of diabetes?
```

```text
What are the risk factors for hypertension?
```

```text
What medications should be offered if hypertension is not controlled on step 2 treatment?
```

```text
What are the symptoms of a heart attack?
```

```text
What should someone do if they think they are having a heart attack?
```

---

# 🧩 Technologies

- Python 3.12
- Streamlit
- LangChain
- FAISS
- BM25
- Sentence Transformers
- CrossEncoder
- PyTorch
- Hugging Face
- Ollama
- Llama 3.1
- PyPDF

---

# 🎯 Project Goals

1. Improve retrieval quality for medical documents.
2. Generate answers grounded in retrieved evidence.
3. Provide source and page-level traceability.
4. Reject unsupported questions.
5. Support dynamic medical PDF ingestion.
6. Combine semantic and keyword retrieval.
7. Provide both local and hosted inference.
8. Build a practical Medical AI / RAG application for research and portfolio use.

---

# 📈 Current Project Results

```text
✅ FAISS Semantic Search
✅ BM25 Keyword Search
✅ Hybrid Retrieval
✅ CrossEncoder Reranking
✅ Grounded Llama 3.1 Generation
✅ Unknown Question Rejection
✅ PDF Knowledge Base Expansion
✅ Source + Page Evidence
✅ Streamlit Interface
✅ Local Ollama Support
✅ Hosted Hugging Face Support
✅ Automated Testing
```

---

# 👩‍💻 Author

## Shahd Fayez

**AI Engineer | Medical AI | Computer Vision | Deep Learning | Machine Learning | NLP**

Built as a portfolio project demonstrating practical experience in:

**Medical AI • Retrieval-Augmented Generation • Information Retrieval • NLP • Generative AI**
