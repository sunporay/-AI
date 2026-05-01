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
        batch_size: int = 2,
        max_length: int = 512,
    ):
        device = f"cuda:{gpu_id}"
        # BUG FIX: was "[BGEEmbeddings]" — typo, class is QwenEmbeddings
        print(f"[QwenEmbeddings] Loading {model_name} on {device}", flush=True)
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Document side: no prefix."""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Query side: prepend BGE instruction prefix."""
        prefixed = f"{BGE_QUERY_PREFIX}{text}"
        return self.model.encode(
            prefixed,
            normalize_embeddings=True,
        ).tolist()


def split_oversized_chunks(docs, max_tokens: int = MAX_CHUNK_TOKENS):
    """Re-split any chunk exceeding max_tokens, preserving original metadata."""
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


# ─── Singleton (per-process) ──────────────────────────────────────────────────
# NOTE: These globals are per-process. In multi-processing / DDP contexts,
# each rank has its own copy. gpu_id should match the rank's assigned GPU.

_global_emb = None
_global_splitter = None
_global_gpu_id: int | None = None


def get_emb_splitter(gpu_id: int):
    """Return (embedding_model, semantic_splitter) for the given GPU.

    On the first call the models are loaded; subsequent calls with the
    *same* gpu_id return the cached objects.  If gpu_id changes (e.g. a
    different rank calls this function in the same process) the models
    are reloaded on the new GPU.
    """
    global _global_emb, _global_splitter, _global_gpu_id
    if _global_emb is None or _global_gpu_id != gpu_id:
        _global_gpu_id = gpu_id
        _global_emb = QwenEmbeddings(gpu_id=gpu_id)
        _global_splitter = SemanticChunker(
            embeddings=_global_emb,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=0.8,
        )
    return _global_emb, _global_splitter