import torch

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )

    def get_embeddings(self):
        return self.embeddings
