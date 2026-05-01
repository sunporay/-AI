import numpy as np
from typing import List
from langchain_core.load import dumps, loads
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
def get_unique_union(documents: list[list]) -> list:
    """將多個檢索結果合併並去重"""
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    unique_docs = list(set(flattened_docs))
    return [loads(doc) for doc in unique_docs]

def create_bm25_index(documents: List[Document]) -> BM25Okapi:
    """建立 BM25 關鍵字檢索索引"""
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)

def fusion_retrieval(vectorstore, bm25: BM25Okapi, splits: List[Document],
                     query: str, k: int = 10, alpha: float = 0.5) -> List[Document]:
    """融合 FAISS 向量與 BM25 關鍵字的分數"""
    epsilon = 1e-8
    all_docs = splits

    bm25_scores = bm25.get_scores(query.split())
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))

    vector_score_map = {doc.page_content: score for doc, score in vector_results}
    vector_scores = np.array([vector_score_map.get(doc.page_content, 0) for doc in all_docs])

    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / \
                    (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    bm25_scores = (bm25_scores - np.min(bm25_scores)) / \
                  (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_scores
    sorted_indices = np.argsort(combined_scores)[::-1]
    return [all_docs[i] for i in sorted_indices[:k]]