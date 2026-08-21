from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)


class DocumentSplitter:
    """
    Splits medical documents into
    overlapping chunks.
    """

    def __init__(
        self,
        chunk_size=700,
        chunk_overlap=150
    ):

        self.text_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,

                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    ""
                ]
            )
        )

    def split_documents(
        self,
        documents
    ):

        return self.text_splitter.split_documents(
            documents
        )