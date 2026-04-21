import json
import os
import re
import uuid
from typing import List

import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import (
    CHUNK_STORE_PATH,
    SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE,
    SEMANTIC_CHUNK_MAX_CHARS,
    SEMANTIC_CHUNK_MAX_SENTENCES,
    SEMANTIC_CHUNK_MIN_SENTENCES,
    VECTOR_DB_PATH,
)
from services.embeddings import get_embedding_model


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# def _split_sentences(text: str) -> List[str]:
#     normalized = text.replace("\x00", " ")
#     sentences = [
#         _normalize_text(sentence)
#         for sentence in SENTENCE_SPLIT_PATTERN.split(normalized)
#         if _normalize_text(sentence)
#     ]
#     return sentences


# def _cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
#     norm_a = np.linalg.norm(vec_a)
#     norm_b = np.linalg.norm(vec_b)

#     if norm_a == 0 or norm_b == 0:
#         return 0.0

#     similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
#     return 1.0 - similarity


# def semantic_chunk_text(text: str, embedding_model) -> List[str]:
#     sentences = _split_sentences(text)

#     if not sentences:
#         return []

#     if len(sentences) <= SEMANTIC_CHUNK_MIN_SENTENCES:
#         return [_normalize_text(" ".join(sentences))]

#     sentence_embeddings = np.array(embedding_model.embed_documents(sentences))
#     distances = [
#         _cosine_distance(sentence_embeddings[index], sentence_embeddings[index + 1])
#         for index in range(len(sentences) - 1)
#     ]

#     breakpoint_threshold = (
#         float(np.percentile(distances, SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE))
#         if distances else 1.0
#     )

#     chunks = []
#     current_chunk = [sentences[0]]

#     for index in range(1, len(sentences)):
#         next_sentence = sentences[index]
#         current_text = " ".join(current_chunk)
#         next_distance = distances[index - 1]

#         should_break = (
#             len(current_chunk) >= SEMANTIC_CHUNK_MIN_SENTENCES
#             and next_distance >= breakpoint_threshold
#         )
#         exceeds_sentence_limit = len(current_chunk) >= SEMANTIC_CHUNK_MAX_SENTENCES
#         exceeds_char_limit = len(current_text) + len(next_sentence) + 1 > SEMANTIC_CHUNK_MAX_CHARS

#         if should_break or exceeds_sentence_limit or exceeds_char_limit:
#             chunks.append(_normalize_text(current_text))
#             current_chunk = [next_sentence]
#             continue

#         current_chunk.append(next_sentence)

#     if current_chunk:
#         chunks.append(_normalize_text(" ".join(current_chunk)))

#     return [chunk for chunk in chunks if chunk]

def semantic_chunk_text(text: str, embedding_model=None) -> List[str]:
    """
    Industry-grade chunking:
    - Structure-aware (paragraph → sentence fallback)
    - Size-controlled
    - Overlap included
    """

    # Step 1: Normalize
    text = _normalize_text(text)

    # Step 2: Recursive splitter (industry standard)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,         
        chunk_overlap=150,       
        separators=[
            "\n\n",              # paragraph
            "\n",                # line
            ". ",                # sentence
            " ",                 # word
        ]
    )

    chunks = splitter.split_text(text)

    # Step 3: Clean chunks
    cleaned_chunks = []
    for chunk in chunks:
        chunk = _normalize_text(chunk)

        # Skip tiny junk chunks
        if len(chunk) < 100:
            continue

        cleaned_chunks.append(chunk)

    return cleaned_chunks


def _load_chunk_store() -> List[dict]:
    if not os.path.exists(CHUNK_STORE_PATH):
        return []

    with open(CHUNK_STORE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def _save_chunk_store(chunks: List[dict]) -> None:
    with open(CHUNK_STORE_PATH, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=True, indent=2)


def _build_documents(chunk_entries: List[dict]) -> List[Document]:
    documents = []

    for entry in chunk_entries:
        metadata = dict(entry["metadata"])
        metadata["chunk_id"] = entry["chunk_id"]
        documents.append(Document(page_content=entry["text"], metadata=metadata))

    return documents


def process_pdf(file_path):
    embedding_model = get_embedding_model()
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    new_chunk_entries = []

    for page_number, page in enumerate(pages, start=1):
        chunks = semantic_chunk_text(page.page_content, embedding_model)

        for chunk_index, chunk_text in enumerate(chunks, start=1):
            new_chunk_entries.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "page": page.metadata.get("page", page_number - 1) + 1,
                    "chunk_index": chunk_index,
                },
            })

    if not new_chunk_entries:
        return {"chunks": 0, "source": os.path.basename(file_path)}

    all_chunk_entries = _load_chunk_store()
    all_chunk_entries.extend(new_chunk_entries)
    _save_chunk_store(all_chunk_entries)

    documents = _build_documents(all_chunk_entries)
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(VECTOR_DB_PATH)

    return {
        "chunks": len(new_chunk_entries),
        "source": os.path.basename(file_path),
    }
