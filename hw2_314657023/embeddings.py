from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL_NAME

MAX_CHUNK_TOKENS = 100

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class QwenEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        gpu_id: int = 0,
        batch_size: int = 4,
        max_length: int = 512,
    ):
        device = f"cuda:{gpu_id}"
        print(f"[BGEEmbeddings] Loading {model_name} on {device}", flush=True)
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Document 端不加 prefix。"""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Query 端加 BGE 官方 prefix。"""
        prefixed = f"{BGE_QUERY_PREFIX}{text}"
        return self.model.encode(
            prefixed,
            normalize_embeddings=True,
        ).tolist()


def split_oversized_chunks(docs, max_tokens: int = MAX_CHUNK_TOKENS):
    """把超過 max_tokens 的 chunk 二次切割，保留原本 metadata。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=10,
        length_function=lambda t: len(t.split()),
    )
    result = []
    for doc in docs:
        token_count = len(doc.page_content.split())
        if token_count > max_tokens:
            sub_docs = splitter.create_documents(
                [doc.page_content],
                metadatas=[doc.metadata],
            )
            result.extend(sub_docs)
        else:
            result.append(doc)
    return result


_global_emb = None
_global_splitter = None


def get_emb_splitter(gpu_id: int):
    global _global_emb, _global_splitter
    if _global_emb is None:
        _global_emb = QwenEmbeddings(gpu_id=gpu_id)
        _global_splitter = SemanticChunker(
            embeddings=_global_emb,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.8,
        )
    return _global_emb, _global_splitter
