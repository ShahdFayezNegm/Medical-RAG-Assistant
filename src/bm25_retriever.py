import re

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Keyword-based retrieval using BM25.
    """

    def __init__(
        self,
        documents
    ):

        self.documents = documents

        self.corpus = [
            self._tokenize(
                doc.page_content
            )
            for doc in documents
        ]

        self.bm25 = BM25Okapi(
            self.corpus
        )

    def _tokenize(
        self,
        text
    ):

        text = text.lower()

        text = re.sub(
            r"[^\w\s]",
            " ",
            text
        )

        return text.split()

    def search(
        self,
        query,
        k=5
    ):

        query_tokens = self._tokenize(
            query
        )

        if not query_tokens:
            return []

        return self.bm25.get_top_n(
            query_tokens,
            self.documents,
            n=k
        )