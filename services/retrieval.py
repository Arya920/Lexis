import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config.settings import (
    CHUNK_STORE_PATH,
    FINAL_TOP_K,
    HYBRID_CANDIDATE_COUNT,
    HYBRID_KEYWORD_WEIGHT,
    HYBRID_SEMANTIC_WEIGHT,
    VECTOR_DB_PATH,
)
from services.embeddings import get_embedding_model, get_reranker_model


TOKEN_PATTERN = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _normalize_scores(score_map: Dict[str, float]) -> Dict[str, float]:
    if not score_map:
        return {}

    values = list(score_map.values())
    minimum = min(values)
    maximum = max(values)

    if math.isclose(minimum, maximum):
        return {key: 1.0 for key in score_map}

    return {
        key: (value - minimum) / (maximum - minimum)
        for key, value in score_map.items()
    }


class BM25Index:
    def __init__(self, chunk_entries: List[dict], k1: float = 1.5, b: float = 0.75):
        self.chunk_entries = chunk_entries
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freq = defaultdict(int)
        self.term_freqs = []
        self.doc_lengths = []
        self.avg_doc_len = 0.0

        for entry in chunk_entries:
            tokens = _tokenize(entry["text"])
            token_counts = Counter(tokens)
            self.documents.append(entry)
            self.term_freqs.append(token_counts)
            self.doc_lengths.append(len(tokens))

            for token in token_counts:
                self.doc_freq[token] += 1

        self.avg_doc_len = (
            sum(self.doc_lengths) / len(self.doc_lengths)
            if self.doc_lengths else 0.0
        )

    def get_scores(self, query: str) -> Dict[str, float]:
        query_tokens = _tokenize(query)
        if not query_tokens or not self.documents:
            return {}

        total_docs = len(self.documents)
        scores = {}

        for index, entry in enumerate(self.documents):
            score = 0.0
            doc_len = self.doc_lengths[index] or 1
            term_freqs = self.term_freqs[index]

            for token in query_tokens:
                frequency = term_freqs.get(token, 0)
                if frequency == 0:
                    continue

                document_frequency = self.doc_freq.get(token, 0)
                idf = math.log(1 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5))
                numerator = frequency * (self.k1 + 1)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * (doc_len / (self.avg_doc_len or 1))
                )
                score += idf * (numerator / denominator)

            if score > 0:
                scores[entry["chunk_id"]] = score

        return scores


def _load_chunk_store() -> List[dict]:
    if not os.path.exists(CHUNK_STORE_PATH):
        return []

    with open(CHUNK_STORE_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def expand_query(query: str) -> str:
    if "tcs" in query.lower():
        return f"{query} Tata Consultancy Services experience projects"
    return query


def _build_chunk_lookup(chunk_entries: List[dict]) -> Dict[str, dict]:
    return {entry["chunk_id"]: entry for entry in chunk_entries}


def retrieve_docs(query, k=FINAL_TOP_K):
    expanded_query = expand_query(query)
    index_path = os.path.join(VECTOR_DB_PATH, "index.faiss")

    if not os.path.exists(index_path):
        return []

    chunk_entries = _load_chunk_store()
    if not chunk_entries:
        return []

    embedding_model = get_embedding_model()
    reranker_model = get_reranker_model()
    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    semantic_hits = db.similarity_search_with_score(expanded_query, k=HYBRID_CANDIDATE_COUNT)
    semantic_scores = {}

    for document, distance in semantic_hits:
        chunk_id = document.metadata.get("chunk_id")
        if not chunk_id:
            continue

        semantic_scores[chunk_id] = 1.0 / (1.0 + float(distance))

    bm25_index = BM25Index(chunk_entries)
    keyword_scores = bm25_index.get_scores(expanded_query)

    normalized_semantic = _normalize_scores(semantic_scores)
    normalized_keyword = _normalize_scores(keyword_scores)

    candidate_ids = set(normalized_semantic) | set(normalized_keyword)
    chunk_lookup = _build_chunk_lookup(chunk_entries)
    candidates = []

    for chunk_id in candidate_ids:
        chunk_entry = chunk_lookup.get(chunk_id)
        if not chunk_entry:
            continue

        hybrid_score = (
            HYBRID_SEMANTIC_WEIGHT * normalized_semantic.get(chunk_id, 0.0)
            + HYBRID_KEYWORD_WEIGHT * normalized_keyword.get(chunk_id, 0.0)
        )

        candidates.append({
            "chunk_id": chunk_id,
            "text": chunk_entry["text"],
            "metadata": chunk_entry["metadata"],
            "hybrid_score": hybrid_score,
            "semantic_score": normalized_semantic.get(chunk_id, 0.0),
            "keyword_score": normalized_keyword.get(chunk_id, 0.0),
        })

    if not candidates:
        return []

    candidates.sort(key=lambda item: item["hybrid_score"], reverse=True)
    rerank_pool = candidates[:HYBRID_CANDIDATE_COUNT]
    rerank_inputs = [(expanded_query, item["text"]) for item in rerank_pool]
    rerank_scores = reranker_model.predict(rerank_inputs)

    final_results = []

    for item, rerank_score in zip(rerank_pool, rerank_scores):
        metadata = dict(item["metadata"])
        metadata.update({
            "chunk_id": item["chunk_id"],
            "hybrid_score": round(float(item["hybrid_score"]), 4),
            "semantic_score": round(float(item["semantic_score"]), 4),
            "keyword_score": round(float(item["keyword_score"]), 4),
            "rerank_score": round(float(rerank_score), 4),
        })
        final_results.append(Document(page_content=item["text"], metadata=metadata))

    final_results.sort(key=lambda document: document.metadata["rerank_score"], reverse=True)
    return final_results[:k]
