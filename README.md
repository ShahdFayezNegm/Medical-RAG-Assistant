# 🩺 Medical RAG Assistant

> **Evidence-grounded medical question answering** using **FAISS + BM25 + CrossEncoder reranking + Llama 3.1**.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/BM25-Keyword%20Retrieval-orange.svg)](https://en.wikipedia.org/wiki/Okapi_BM25)
[![Llama](https://img.shields.io/badge/Llama%203.1-8B-purple.svg)](https://www.llama.com/)

---

## 🚀 Live Demo

**Try the Medical RAG Assistant directly in your browser:**

👉 [Open the Live Demo](https://medical-rag-assistant-tulcwbelmhazmjdu4pjw4p.streamlit.app/)

No local setup is required to use the hosted application.

---

## 📌 Overview

Medical RAG Assistant is a **Retrieval-Augmented Generation (RAG)** system for evidence-grounded question answering over an indexed collection of medical PDF documents.

Instead of relying only on the language model's internal knowledge, the system first retrieves relevant passages from the medical knowledge base, reranks them, and then generates an answer using the selected evidence.

The application displays the supporting documents and page numbers used for the answer, providing **source and page-level traceability**.

It also supports adding new medical PDFs to the existing knowledge base through the Streamlit interface.

> **Scope:** This project is designed for research, educational, and portfolio purposes. It is an evidence-grounded medical question-answering system rather than an autonomous clinical diagnosis or treatment recommendation system.

---

# ✨ Key Features

* 📚 **Medical PDF Knowledge Base** — search an indexed collection of medical documents.
* 🔎 **Hybrid Retrieval** — combines FAISS semantic retrieval with BM25 keyword retrieval.
* 🎯 **CrossEncoder Reranking** — reranks retrieved chunks according to question-document relevance.
* 🧠 **Hybrid Relevance Scoring** — combines retrieval and reranking signals before final evidence selection.
* 🛡️ **Evidence-Grounded Generation** — instructs the LLM to answer only from retrieved evidence.
* 📖 **Supporting Evidence** — displays source documents, page numbers, and retrieved passages.
* ➕ **Incremental PDF Ingestion** — add new medical PDFs directly through the Streamlit interface.
* 📊 **Knowledge Base Statistics** — displays document and chunk counts.
* ⏱️ **Performance Tracking** — reports retrieval, generation, and total response time.
* 🚫 **Unknown Question Rejection** — unsupported questions can be rejected instead of generating unsupported answers.
* 💻 **Local Inference** — Ollama + Llama 3.1 8B.
* ☁️ **Hosted Inference** — Hugging Face Inference Providers for deployment.
* 🧪 **Automated Testing** — tests document loading, chunking, retrieval, indexing, PDF ingestion, RAG, and evaluation.

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
                         │ Relevance Filtering  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     Llama 3.1 8B     │
                         │ Evidence-grounded    │
                         │     Generation       │
                         └──────────┬───────────┘
                                    │
                                    ▼
             ┌─────────────────────────────────────────┐
             │ Answer + Evidence + Sources + Timing    │
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
    ├── test_mmr.py
    ├── test_rag.py
    ├── test_rag_multiple.py
    ├── test_retrievers.py
    ├── test_splitter.py
    └── test_vector_store.py
```

> **Security:** `.streamlit/secrets.toml` is used locally for secrets such as `HF_TOKEN` but is intentionally excluded from GitHub.

---

# 🔧 Technology Stack

| Component            | Technology                                  |
| -------------------- | ------------------------------------------- |
| Programming Language | Python 3.12                                 |
| User Interface       | Streamlit                                   |
| Semantic Retrieval   | FAISS                                       |
| Keyword Retrieval    | BM25                                        |
| Reranking            | Sentence-Transformers CrossEncoder          |
| Embeddings           | BAAI/bge-small-en-v1.5                      |
| LLM                  | Llama 3.1 8B                                |
| Local Inference      | Ollama                                      |
| Hosted Inference     | Hugging Face Inference Providers            |
| PDF Processing       | PyMuPDF + LangChain Community Loaders       |
| Framework            | LangChain                                   |
| Vector Store         | FAISS                                       |
| Evaluation           | Hit@1, Hit@3, Hit@5, MRR, Unknown Rejection |

---

# 🔎 Retrieval Pipeline

## 1. FAISS Semantic Retrieval

FAISS provides vector-based semantic retrieval.

This allows the system to retrieve relevant passages even when the wording of the question differs from the wording used in the source document.

```text
Question
   ↓
Embedding
   ↓
Vector Similarity Search
   ↓
Relevant Chunks
```

---

## 2. BM25 Keyword Retrieval

BM25 provides lexical keyword-based retrieval.

It is particularly useful for:

* Medical terminology
* Drug names
* Clinical terms
* Exact phrases
* Guideline terminology
* Recommendation numbers

```text
Question
   ↓
Token Matching
   ↓
BM25 Ranking
   ↓
Relevant Chunks
```

---

## 3. Hybrid Retrieval

The system combines FAISS semantic retrieval and BM25 keyword retrieval.

The results are merged and deduplicated before reranking.

```text
Semantic Search
       +
Keyword Search
       ↓
Hybrid Retrieval
```

This combination helps balance semantic similarity with exact medical terminology matching.

---

## 4. CrossEncoder Reranking

A CrossEncoder evaluates the relevance between the user question and each retrieved chunk.

The merged candidate set is reranked before the final evidence is selected.

```text
Hybrid Candidates
       ↓
CrossEncoder
       ↓
Relevance Ranking
       ↓
Top Evidence
```

---

## 5. Relevance Filtering

After reranking, the system applies relevance signals to select the evidence used for generation.

The final context is limited to selected high-relevance chunks to reduce unnecessary context and improve grounding.

---

## 6. Grounded Generation

Only the selected evidence is passed to Llama 3.1.

The generator is instructed to:

```text
Use ONLY the provided evidence.
Do not use outside medical knowledge.
Do not guess.
Do not speculate.
Do not add unsupported medical facts.
Preserve conditions and qualifiers.
```

When the evidence is insufficient, the system can return:

```text
I don't have enough information.
```

This design aims to reduce unsupported generation and improve traceability.

---

# 📚 Knowledge Base

The application supports a growing medical PDF knowledge base.

Current hosted deployment statistics observed during testing:

```text
Documents: 301
Chunks:    913
```

The knowledge base includes medical guidelines and other permitted medical reference documents.

---

# ➕ Incremental PDF Ingestion

A new medical PDF can be added through the Streamlit interface.

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
FAISS Update
     ↓
BM25 Update
     ↓
Knowledge Base Updated
```

This allows new documents to become searchable without rebuilding the entire application manually.

---

# ✅ Example

## Question

```text
What is the first-line treatment for type 2 diabetes?
```

## Grounded Answer

The application generates an answer from the retrieved evidence and displays the supporting source and page.

Example retrieved evidence:

```text
Source: data/diabetes_guideline.pdf
Page: 68
```

Additional supporting evidence may also be displayed when relevant.

> The generated answer is based on the evidence retrieved from the indexed medical documents.

---

# ❤️ New Medical Document Test

An additional NHS heart-attack document was added through the application.

Example question:

```text
What are the symptoms of a heart attack?
```

The system successfully retrieved evidence from the newly indexed document and displayed the corresponding source and page information.

This demonstrates the complete ingestion-to-answer workflow:

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

The current retrieval and rejection evaluation achieved:

| Metric            |      Result |
| ----------------- | ----------: |
| Hit@1             |  **75.00%** |
| Hit@3             | **100.00%** |
| Hit@5             | **100.00%** |
| MRR               |   **0.875** |
| Unknown Rejection | **100.00%** |

These values are based on the project's current evaluation set and represent a **project benchmark**, not clinical validation.

---

# 📈 Evaluation Metrics

## Hit@1

Percentage of evaluation questions where the expected source page is ranked first.

## Hit@3

Percentage of evaluation questions where the expected source page appears within the top three results.

## Hit@5

Percentage of evaluation questions where the expected source page appears within the top five results.

## MRR

Mean Reciprocal Rank measures how early the correct evidence appears in the retrieval ranking.

## Unknown Rejection

Measures whether unsupported questions are rejected rather than receiving an unsupported answer.

---

# 🛡️ Grounding Strategy

The generator is explicitly instructed to:

* Use only the retrieved evidence.
* Avoid outside medical knowledge.
* Avoid unsupported inference.
* Preserve treatment conditions and qualifiers.
* Avoid inventing sources or page numbers.
* State when the available evidence is insufficient.

Unsupported questions can return:

```text
I don't have enough information.
```

The system is designed as an **evidence-grounded medical question-answering assistant**, rather than an autonomous clinical decision-making system.

---

# ⚡ Performance

The application tracks:

* Retrieval time
* Generation time
* Total response time

Example hosted deployment result:

```text
Retrieval:   1.48 seconds
Generation:  2.58 seconds
Total:       4.05 seconds
```

Performance can vary depending on hosted inference availability, document size, retrieval workload, and infrastructure.

Local Ollama performance can also vary significantly depending on hardware.

---

# 🤖 LLM Modes

## Local Mode

```text
User
 ↓
Streamlit
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
Streamlit Community Cloud
 ↓
RAG Pipeline
 ↓
Hugging Face Inference Providers
 ↓
Llama 3.1 8B Instruct
```

When `HF_TOKEN` is configured, the generator uses hosted Hugging Face inference.

The local Ollama path remains available for local development.

---

# 🖥️ Streamlit Interface

The application provides:

* 📚 Medical PDF upload
* 📊 Knowledge base statistics
* 💬 Medical question input
* 🧠 Evidence-grounded answers
* ⏱️ Retrieval and generation timing
* 📖 Supporting evidence
* 📄 Source documents
* 🔢 Page numbers
* 🔬 RAG pipeline details

---

# 🚀 Local Setup

## 1. Clone the Repository

```bash
git clone https://github.com/ShahdFayezNegm/Medical-RAG-Assistant.git
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

Verify the model:

```powershell
ollama list
```

## 5. Build the FAISS Index

If the index needs to be rebuilt:

```powershell
python build_faiss.py
```

## 6. Run Evaluation

```powershell
python tests/test_evaluation.py
```

Expected benchmark for the tested evaluation set:

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

The hosted deployment uses:

```text
GitHub
   ↓
Streamlit Community Cloud
   ↓
Medical RAG Pipeline
   ↓
Hugging Face Inference Providers
   ↓
Llama 3.1 8B Instruct
```

### Streamlit Secret

Configure:

```toml
HF_TOKEN = "hf_your_token_here"
```

The real token should **never** be committed to GitHub.

### Current Live Demo

👉 [Medical RAG Assistant](https://medical-rag-assistant-tulcwbelmhazmjdu4pjw4p.streamlit.app/)

---

# 🔐 Security & Privacy

Never commit:

```text
.venv/
__pycache__/
.env
.streamlit/secrets.toml
API keys
Private credentials
Private patient information
Private medical documents
Private model weights
```

For a public medical-document demonstration, only use documents that you are permitted to redistribute.

Do not upload personally identifiable patient information.

---

# 🧪 Testing

The project contains tests covering document ingestion, preprocessing, retrieval, indexing, and RAG behavior.

## Test Suite

| Test                   | Purpose                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `test_loader.py`       | Tests PDF loading and document processing                  |
| `test_splitter.py`     | Tests document chunking and splitting                      |
| `test_vector_store.py` | Tests FAISS vector-store creation/loading                  |
| `test_bm25.py`         | Tests BM25 keyword retrieval                               |
| `test_retrievers.py`   | Tests retrieval components                                 |
| `test_mmr.py`          | Tests MMR-based retrieval                                  |
| `test_add_pdf.py`      | Tests adding a new PDF                                     |
| `test_rag.py`          | Tests the RAG pipeline                                     |
| `test_rag_multiple.py` | Tests RAG behavior across multiple questions               |
| `test_evaluation.py`   | Evaluates retrieval ranking and unknown-question rejection |

---

## Run Individual Tests

```powershell
python tests/test_loader.py
python tests/test_splitter.py
python tests/test_vector_store.py
python tests/test_bm25.py
python tests/test_retrievers.py
python tests/test_mmr.py
python tests/test_add_pdf.py
python tests/test_rag.py
python tests/test_rag_multiple.py
python tests/test_evaluation.py
```

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

* Python 3.12
* Streamlit
* LangChain
* FAISS
* BM25
* Sentence Transformers
* CrossEncoder
* PyTorch
* Hugging Face
* Ollama
* Llama 3.1
* PyMuPDF
* PyPDF

---

# 🎯 Project Goals

1. Improve retrieval quality for medical documents.
2. Generate answers grounded in retrieved evidence.
3. Provide source and page-level traceability.
4. Reject unsupported questions.
5. Support dynamic medical PDF ingestion.
6. Combine semantic and keyword retrieval.
7. Provide both local and hosted inference.
8. Demonstrate a practical Medical AI / RAG application for research and portfolio use.

---

# 📈 Current Project Results

```text
✅ FAISS Semantic Retrieval
✅ BM25 Keyword Retrieval
✅ Hybrid Retrieval
✅ CrossEncoder Reranking
✅ Evidence-Grounded Llama 3.1 Generation
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

Interested in building practical AI systems combining:

**Medical AI • Retrieval-Augmented Generation • Deep Learning • NLP • Computer Vision • Evidence-Grounded AI**

