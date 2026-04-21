from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import logging

from config.settings import EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME

logging.getLogger("transformers").setLevel(logging.ERROR)

_embedding_model = None
_reranker_model = None

def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )

    return _embedding_model


def get_reranker_model():
    global _reranker_model

    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)

    return _reranker_model
