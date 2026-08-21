class Retriever:
    """
    FAISS Retriever supporting
    similarity search and MMR search.
    """

    def __init__(
        self,
        vector_db
    ):
        self.db = vector_db

    def similarity_search(
        self,
        query,
        k=5
    ):

        return self.db.similarity_search(
            query,
            k=k
        )

    def mmr_search(
        self,
        query,
        k=5,
        fetch_k=15
    ):

        return self.db.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k
        )