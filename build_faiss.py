import sys
from pathlib import Path


# =========================================================
# Add project root to Python path
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

sys.path.append(
    str(PROJECT_ROOT)
)


from src.loader import PDFLoader
from src.splitter import DocumentSplitter
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


FAISS_PATH = "models/faiss_index"


print("=" * 60)
print("BUILDING FAISS INDEX")
print("=" * 60)


# =========================================================
# 1. Load PDFs
# =========================================================

loader = PDFLoader("data")

documents = loader.load_documents()


# =========================================================
# 2. Split documents
# =========================================================

splitter = DocumentSplitter()

chunks = splitter.split_documents(
    documents
)


# =========================================================
# 3. Load embedding model
# =========================================================

embedding_model = EmbeddingModel()

embeddings = (
    embedding_model.get_embeddings()
)


# =========================================================
# 4. Create FAISS
# =========================================================

vector_store = VectorStore(
    embeddings
)

db = vector_store.create(
    chunks
)


# =========================================================
# 5. Save FAISS
# =========================================================

vector_store.save(
    db,
    FAISS_PATH
)


print()
print("=" * 60)
print("FAISS INDEX CREATED SUCCESSFULLY")
print("=" * 60)

print(
    f"Documents: {len(documents)}"
)

print(
    f"Chunks: {len(chunks)}"
)

print(
    f"Saved to: {FAISS_PATH}"
)

print("=" * 60)
