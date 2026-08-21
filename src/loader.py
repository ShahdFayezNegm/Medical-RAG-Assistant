from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    PyPDFDirectoryLoader
)

from src.cleaner import DocumentCleaner


class PDFLoader:
    """
    Loads and cleans PDF documents.

    Supports:
    1. Loading all PDFs from a directory.
    2. Loading one specific PDF file.
    """

    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.cleaner = DocumentCleaner()

    # =========================================================
    # LOAD DIRECTORY
    # =========================================================

    def load_documents(self):

        path = Path(self.data_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Path not found: {path}"
            )

        # If a single PDF was provided
        if path.is_file():

            if path.suffix.lower() != ".pdf":
                raise ValueError(
                    "The provided file is not a PDF."
                )

            return self._load_single_pdf(
                path
            )

        # Otherwise load all PDFs in directory
        loader = PyPDFDirectoryLoader(
            str(path)
        )

        documents = loader.load()

        return self._clean_documents(
            documents
        )

    # =========================================================
    # LOAD ONE PDF
    # =========================================================

    def load_pdf(self, pdf_path):

        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(
                f"PDF not found: {path}"
            )

        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF files are supported."
            )

        return self._load_single_pdf(
            path
        )

    # =========================================================
    # INTERNAL SINGLE PDF LOADER
    # =========================================================

    def _load_single_pdf(self, path):

        loader = PyPDFLoader(
            str(path)
        )

        documents = loader.load()

        return self._clean_documents(
            documents
        )

    # =========================================================
    # CLEAN DOCUMENTS
    # =========================================================

    def _clean_documents(
        self,
        documents
    ):

        cleaned_documents = []

        for doc in documents:

            if self.cleaner.should_skip(
                doc
            ):
                continue

            doc.page_content = (
                self.cleaner.clean_text(
                    doc.page_content
                )
            )

            cleaned_documents.append(
                doc
            )

        print("=" * 50)

        print(
            f"Original documents : "
            f"{len(documents)}"
        )

        print(
            f"Cleaned documents  : "
            f"{len(cleaned_documents)}"
        )

        print("=" * 50)

        return cleaned_documents