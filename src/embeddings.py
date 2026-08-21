import os
import torch

# Disable online model downloads if you intentionally want offline mode.
# NOTE: Streamlit Cloud needs the model to be available in its environment.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:

    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Using embedding device: {device}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={
                "device": device
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    def get_embeddings(self):
        return self.embeddings
