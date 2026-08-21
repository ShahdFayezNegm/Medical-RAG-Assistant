from langchain_community.vectorstores import FAISS


class VectorStore:
    """
    Creates, saves, and loads FAISS vector stores.
    """

    def __init__(
        self,
        embeddings
    ):
        self.embeddings = embeddings

    # =========================================================
    # CREATE
    # =========================================================

    def create(
        self,
        chunks
    ):

        return FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

    # =========================================================
    # SAVE
    # =========================================================

    def save(
        self,
        vector_store,
        path="models/faiss_index"
    ):

        vector_store.save_local(
            path
        )

    # =========================================================
    # LOAD
    # =========================================================

    def load(
        self,
        path="models/faiss_index"
    ):

        return FAISS.load_local(
            folder_path=path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )

    # =========================================================
    # RETRIEVER
    # =========================================================

    def as_retriever(
        self,
        vector_store,
        search_type="mmr",
        k=5,
        fetch_k=20
    ):

        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k
            }
        )