import sys
from pathlib import Path


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


from src.loader import PDFLoader
from src.splitter import DocumentSplitter
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


# =========================================================
# Load Documents
# =========================================================

loader = PDFLoader("data")

documents = loader.load_documents()


# =========================================================
# Split Documents
# =========================================================

splitter = DocumentSplitter()

chunks = splitter.split_documents(
    documents
)


# =========================================================
# Embeddings
# =========================================================

embedding_model = EmbeddingModel()

embeddings = (
    embedding_model.get_embeddings()
)


# =========================================================
# Create FAISS
# =========================================================

vector_store = VectorStore(
    embeddings
)

db = vector_store.create(
    chunks
)


# =========================================================
# Results
# =========================================================

print("=" * 60)
print("VECTOR STORE TEST")
print("=" * 60)

print(
    "Documents:",
    len(documents)
)

print(
    "Chunks:",
    len(chunks)
)

print()
print(
    "FAISS vector store created successfully."
)

print()
print(
    "Vector store type:",
    type(db).__name__
)

print()
print("Vector store test completed.")