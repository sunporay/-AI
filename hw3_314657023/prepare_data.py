"""prepare_data.py — query rewrite, retrieval, rerank, and prompt-cache generation.

GPU layout (configurable via env vars)
───────────────────────────────────────
  GPU_EMB     (default 0) : embedding model
  GPU_RERANK  (default 1) : reranker model
  GPU_CHAT    (default 2, fallback to GPU_INFER) : chat model (used only for GPU_REWRITE default)
  GPU_REWRITE (default GPU_CHAT) : query rewrite model

During cmd_prepare_data:
  • Query rewrite → retrieval → rerank → prompt cache JSON written to disk.
  • Cached results are read directly by cmd_train and cmd_infer in train.py.

IMPORTANT (chat template):
  build_prompt / build_full_example use **Qwen ChatML** format
  (<|im_start|> ... <|im_end|>) because we fine-tune Qwen2.5-3B-Instruct.
  Do NOT use Llama 3 tags (<|begin_of_text|>, <|start_header_id|>, <|eot_id|>)
  here — Qwen's tokenizer does not recognise them as special tokens and will
  split them into regular sub-tokens, which ruins both training and inference.
"""

import os
import json
import pickle
import re
import torch
from tqdm import tqdm
from langchain_community.vectorstores import FAISS

from embeddings import get_emb_splitter
from reranker import OfficialQwenReranker
from retriever import fusion_retrieval, get_unique_union


# ─── GPU layout constants ─────────────────────────────────────────────────────

GPU_EMB     = int(os.environ.get("GPU_EMB", 0))
GPU_RERANK  = int(os.environ.get("GPU_RERANK", 7))
GPU_CHAT    = int(os.environ.get("GPU_CHAT", os.environ.get("GPU_INFER", 6)))
GPU_REWRITE = int(os.environ.get("GPU_REWRITE", GPU_CHAT))
RERANK_BATCH_SIZE = int(os.environ.get("RERANK_BATCH_SIZE", 2))
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", 2048))
DEFAULT_REWRITE_MODEL = os.environ.get(
    "QUERY_REWRITE_MODEL",
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
)


# ─── Label mapping ────────────────────────────────────────────────────────────

LABEL2ID = {
    "Attribution Failure": 0,
    "Entity":              1,
    "Number":              2,
    "Overgeneralization":  3,
    "Temporal":            4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def normalize_label(raw: str) -> str:
    raw = raw.strip()
    if raw in LABEL2ID:
        return raw
    if raw.isdigit() and int(raw) in ID2LABEL:
        return ID2LABEL[int(raw)]
    return raw


# ─── Prompt templates (Qwen ChatML) ───────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a hallucination detection assistant.

Your task:
Determine whether the claim is supported by the context.

Definition:
A hallucination is any part of the answer that is not directly supported by the provided context.

This includes:
- Attribution Failure: A claim lacks proper attribution, either crediting the wrong source or presenting information as fact without citation.
- Entity: A claim includes swapped, incorrectly specified, or inserted noun phrases (e.g. one named entity used in a context where another word is expected).
- Number: A claim has a different number than the original context (e.g. 20% vs. 0.7%). Any number, including year, dimensions, ages, etc.
- Overgeneralization: A claim is based on accurate contextual information but is too broad or too general to be supported by the context.
- Temporal: A claim does not accurately incorporate tense, modality (e.g. might vs. will), or time reference in relation to the context.

Instruction:
Based on the context and evidence provided, output ONLY one of the following labels (no explanation):
Attribution Failure, Entity, Number, Overgeneralization, Temporal"""

# Regex to strip Qwen ChatML special tokens from decoded model output.
# Used after tokenizer.decode(skip_special_tokens=True) as a belt-and-braces
# cleanup in case any slipped through.
QWEN_CHATML_PATTERN = re.compile(
    r"<\|im_start\|>(?:user|assistant|system)?\n?|<\|im_end\|>\n?|<\|endoftext\|>"
)


def build_prompt(claim: str, context: str) -> str:
    """Build a Qwen ChatML prompt (no answer yet; assistant turn is open).

    Ends with '<|im_start|>assistant\\n' so the model generates the label next.
    Note: train.py's RESPONSE_TEMPLATE matches on '<|im_start|>assistant',
    so this exact prefix must appear verbatim in the training text.
    """
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Original Claim:\n{claim}\n\n"
        f"Context (retrieved via expanded queries):\n{context}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def build_full_example(claim: str, context: str, label: str) -> str:
    """Build a complete training example (prompt + answer + <|im_end|>)."""
    return build_prompt(claim, context) + label + "<|im_end|>"


# ─── Query rewriter setup ─────────────────────────────────────────────────────

def setup_query_rewriter(
    model_name: str = DEFAULT_REWRITE_MODEL,
    gpu_id: int = GPU_REWRITE,
    max_new_tokens: int = 128,
):
    """Load the query rewriter model. Falls back to original-claim-only mode."""
    try:
        from transformers import pipeline as hf_pipeline
        from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
        from query_rewrite import QueryRewriter

        print(f"[INFO] Loading query rewriter on GPU {gpu_id}: {model_name}")
        pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            device=f"cuda:{gpu_id}",
            torch_dtype=torch.float16,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
        )
        llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))
        return QueryRewriter(llm=llm)
    except Exception as e:
        print(f"[WARN] Query rewriter unavailable on GPU {gpu_id}: {e}")
        print("[WARN] Falling back to original claim only (no query expansion).")
        return None


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def expand_queries(
    claim: str,
    query_rewriter=None,
    max_queries: int = 3,
) -> list[str]:
    queries = [claim]
    if query_rewriter is not None:
        try:
            queries = query_rewriter.generate_queries(claim)
        except Exception as e:
            print(f"[WARN] query rewrite failed; using original claim only: {e}")
            queries = [claim]

    queries = _dedupe_keep_order(queries)
    if claim not in queries:
        queries.insert(0, claim)
    if max_queries > 0:
        queries = queries[:max_queries]
    return queries


# ─── RAG: load FAISS index and retrieve context ───────────────────────────────

def load_paper_index(paper_id: str, persist_dir: str, embedding_model):
    index_path  = os.path.join(persist_dir, paper_id)
    bm25_path   = os.path.join(index_path, "bm25.pkl")
    splits_path = os.path.join(index_path, "splits.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")

    vectorstore = FAISS.load_local(
        index_path, embedding_model,
        allow_dangerous_deserialization=True,
    )
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)
    with open(splits_path, "rb") as f:
        splits = pickle.load(f)

    return splits, vectorstore, bm25


def retrieve_context(
    claim: str,
    paper_id: str,
    persist_dir: str,
    embedding_model,
    reranker,
    query_rewriter=None,
    index_cache: dict | None = None,
    max_queries: int = 3,
    fusion_k: int = 40,
    alpha: float = 0.5,
    rerank_top_k: int = 4,
) -> dict:
    try:
        if index_cache is not None and paper_id in index_cache:
            splits, vectorstore, bm25 = index_cache[paper_id]
        else:
            splits, vectorstore, bm25 = load_paper_index(
                paper_id, persist_dir, embedding_model
            )
            if index_cache is not None:
                index_cache[paper_id] = (splits, vectorstore, bm25)

        queries = expand_queries(
            claim,
            query_rewriter=query_rewriter,
            max_queries=max_queries,
        )
        all_retrieval = [
            fusion_retrieval(
                vectorstore, bm25, splits, query,
                k=fusion_k, alpha=alpha,
            )
            for query in queries
        ]
        candidates = get_unique_union(all_retrieval)
        if not candidates:
            return {"queries": queries, "context": "", "evidence": []}

        pairs  = [[claim, d.page_content] for d in candidates]
        scores = reranker.predict(pairs)
        scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        docs   = [d for d, _ in scored[:rerank_top_k]]
        return {
            "queries": queries,
            "context": "\n\n".join(d.page_content for d in docs),
            "evidence": [d.page_content for d in docs],
        }
    except Exception as e:
        print(f"[WARN] retrieve_context failed for paper_id='{paper_id}': {e}")
        return {"queries": [claim], "context": "", "evidence": []}


# ─── CSV loading ──────────────────────────────────────────────────────────────

def load_csv(csv_path: str, has_label: bool) -> list[dict]:
    import pandas as pd
    df = pd.read_csv(csv_path)
    split = os.path.splitext(os.path.basename(csv_path))[0]
    df["paper_id"] = df["paper_id"].apply(
        lambda x: x if "/" in str(x) else f"{split}/{x}"
    )
    records = []
    for _, row in df.iterrows():
        rec = {
            "id":       str(row["id"]),
            "paper_id": str(row["paper_id"]),
            "claim":    str(row["text"]),
        }
        if has_label:
            rec["label"] = normalize_label(str(row["label"]))
        records.append(rec)
    return records


# ─── Example preparation ──────────────────────────────────────────────────────

def prepare_examples(
    records: list[dict],
    persist_dir: str,
    embedding_model,
    reranker,
    has_label: bool,
    query_rewriter=None,
    desc: str = "Preparing",
    index_cache: dict | None = None,
    max_queries: int = 3,
    fusion_k: int = 40,
    alpha: float = 0.5,
    rerank_top_k: int = 4,
) -> list[dict]:
    examples = []
    for r in tqdm(records, desc=desc):
        retrieval = retrieve_context(
            claim           = r["claim"],
            paper_id        = r["paper_id"],
            persist_dir     = persist_dir,
            embedding_model = embedding_model,
            reranker        = reranker,
            query_rewriter  = query_rewriter,
            index_cache     = index_cache,
            max_queries     = max_queries,
            fusion_k        = fusion_k,
            alpha           = alpha,
            rerank_top_k    = rerank_top_k,
        )
        context = retrieval["context"]
        ex = {
            "id":       r["id"],
            "paper_id": r["paper_id"],
            "claim":    r["claim"],
            "queries":  retrieval["queries"],
            "context":  context,
            "evidence": retrieval["evidence"],
            "prompt":   build_prompt(r["claim"], context),
        }
        if has_label:
            ex["text"]  = build_full_example(r["claim"], context, r["label"])
            ex["label"] = r["label"]
        examples.append(ex)
    return examples


def prepare_examples_cached(
    records: list[dict],
    persist_dir: str,
    embedding_model,
    reranker,
    has_label: bool,
    desc: str,
    cache_path: str,
    query_rewriter=None,
    index_cache: dict | None = None,
    max_queries: int = 3,
    fusion_k: int = 40,
    alpha: float = 0.5,
    rerank_top_k: int = 4,
) -> list[dict]:
    """Return cached examples if they exist; otherwise run RAG and write cache."""
    if os.path.exists(cache_path):
        print(f"[INFO] Cache hit → {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    examples = prepare_examples(
        records,
        persist_dir,
        embedding_model,
        reranker,
        query_rewriter  = query_rewriter,
        has_label       = has_label,
        desc            = desc,
        index_cache     = index_cache,
        max_queries     = max_queries,
        fusion_k        = fusion_k,
        alpha           = alpha,
        rerank_top_k    = rerank_top_k,
    )
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Cache saved → {cache_path}")
    return examples


def prepare_examples_from_csv(
    csv_path: str,
    persist_dir: str,
    embedding_model,
    reranker,
    query_rewriter,
    has_label: bool,
    cache_path: str,
    desc: str,
    index_cache: dict | None = None,
    max_queries: int = 3,
    fusion_k: int = 40,
    alpha: float = 0.5,
    rerank_top_k: int = 4,
) -> list[dict]:
    records = load_csv(csv_path, has_label=has_label)
    return prepare_examples_cached(
        records         = records,
        persist_dir     = persist_dir,
        embedding_model = embedding_model,
        reranker        = reranker,
        query_rewriter  = query_rewriter,
        has_label       = has_label,
        desc            = desc,
        cache_path      = cache_path,
        index_cache     = index_cache,
        max_queries     = max_queries,
        fusion_k        = fusion_k,
        alpha           = alpha,
        rerank_top_k    = rerank_top_k,
    )


def _load_cache(cache_path: str) -> list[dict]:
    """Unconditionally read a JSON cache."""
    with open(cache_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_examples_path(path: str) -> str:
    return os.path.abspath(path)


def prepare_split_caches(args, include_test: bool = False) -> dict[str, str]:
    persist_dir = os.path.abspath(args.persist_dir)
    output_dir  = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    cache_paths = {
        "train": resolve_examples_path(args.train_examples),
        "dev":   resolve_examples_path(args.dev_examples),
    }
    if include_test:
        cache_paths["test"] = resolve_examples_path(args.test_examples)

    for path in cache_paths.values():
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if getattr(args, "force", False):
        for path in cache_paths.values():
            if os.path.exists(path):
                os.remove(path)
                print(f"[FORCE] removed {path}")

    if all(os.path.exists(path) for path in cache_paths.values()):
        for split, path in cache_paths.items():
            print(f"[INFO] Cache hit ({split}) → {path}")
        return cache_paths

    print(
        f"[INFO] Loading embedding on GPU {GPU_EMB}, "
        f"query rewriter on GPU {GPU_REWRITE}, "
        f"reranker on GPU {GPU_RERANK} "
        f"(batch_size={RERANK_BATCH_SIZE}, max_length={RERANK_MAX_LENGTH}) …"
    )
    emb, _ = get_emb_splitter(gpu_id=GPU_EMB)
    query_rewriter = setup_query_rewriter(
        model_name      = args.rewrite_model,
        gpu_id          = GPU_REWRITE,
        max_new_tokens  = args.rewrite_max_new_tokens,
    )
    reranker = OfficialQwenReranker(
        gpu_id      = GPU_RERANK,
        batch_size  = RERANK_BATCH_SIZE,
        max_length  = RERANK_MAX_LENGTH,
    )

    index_cache: dict = {}
    try:
        for split, csv_attr, label_flag, desc_str in [
            ("train", "train_csv", True,  "Preparing train"),
            ("dev",   "dev_csv",   True,  "Preparing dev"),
            *([("test", "test_csv", False, "Preparing test")] if include_test else []),
        ]:
            prepare_examples_from_csv(
                csv_path        = getattr(args, csv_attr),
                persist_dir     = persist_dir,
                embedding_model = emb,
                reranker        = reranker,
                query_rewriter  = query_rewriter,
                has_label       = label_flag,
                cache_path      = cache_paths[split],
                desc            = desc_str,
                index_cache     = index_cache,
                max_queries     = args.max_queries,
                fusion_k        = args.fusion_k,
                alpha           = args.alpha,
                rerank_top_k    = args.rerank_top_k,
            )
    finally:
        del emb, reranker
        if query_rewriter is not None:
            del query_rewriter
        torch.cuda.empty_cache()

    return cache_paths


# ─── CLI entry point ──────────────────────────────────────────────────────────

def cmd_prepare_data(args):
    cache_paths = prepare_split_caches(args, include_test=args.include_test)

    print("[INFO] Prepared example caches:")
    print(f"   train → {cache_paths['train']}")
    print(f"   dev   → {cache_paths['dev']}")
    if args.include_test:
        print(f"   test  → {cache_paths['test']}")