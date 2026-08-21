import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:

    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",

            model_kwargs={
                "device": "cuda"
            },

            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    def get_embeddings(self):
        return self.embeddings