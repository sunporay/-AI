import torch
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

from config import NUM_GPUS
from embeddings import get_emb_splitter
from pipeline import PaperRAGPipeline
from reranker import rerank_via_service
from retriever import fusion_retrieval, get_unique_union


def query_worker(
    worker_idx: int,
    papers_chunk: list,
    request_queue,
    response_queue,
    result_queue,
    persist_dir: str,
):
    gpu_id = worker_idx % NUM_GPUS
    print(f"[Worker {worker_idx}] Using cuda:{gpu_id}", flush=True)

    embedding_model, _ = get_emb_splitter(gpu_id)

    pipe = hf_pipeline(
        "text-generation",
        model="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        device=f"cuda:{gpu_id}",          # ← 修正：原本寫死 cuda:1
        torch_dtype=torch.float16,         # ← 修正：dtype 改成 torch_dtype
        max_new_tokens=512,
        return_full_text=False,
    )
    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

    rag_pipeline = PaperRAGPipeline(
        embedding_model=embedding_model,
        llm=llm,
        alpha=0.5,
        fusion_k=10,
        rerank_top_k=3,
        num_queries=3,
        persist_dir=persist_dir,
    )

    results = []
    for paper in tqdm(papers_chunk, desc=f"Worker {worker_idx}", position=worker_idx, leave=True):
        original_idx = paper["_original_idx"]
        title     = paper.get("title", f"paper_{original_idx}")
        question  = paper["question"]
        full_text = paper["full_text"]

        try:
            print(f"[Worker {worker_idx}] step1: build_index '{title[:30]}'", flush=True)
            # ← 修正：_build_index 可能不回傳 tuple，改用 persist 再 load 的方式
            rag_pipeline._build_index(full_text, title)
            splits, vectorstore, bm25 = rag_pipeline._load_index(title)

            print(f"[Worker {worker_idx}] step2: generate_queries", flush=True)
            # ← 修正：加 hasattr 防止 mq_expander 未初始化時 crash
            if not hasattr(rag_pipeline, "mq_expander") or rag_pipeline.mq_expander is None:
                raise AttributeError("rag_pipeline.mq_expander is not initialized")
            queries = rag_pipeline.mq_expander.generate_queries(question)

            print(f"[Worker {worker_idx}] step3: retrieval", flush=True)
            all_retrieval = [
                fusion_retrieval(
                    vectorstore, bm25, splits, q,
                    k=rag_pipeline.fusion_k, alpha=rag_pipeline.alpha,
                )
                for q in queries
            ]
            candidates = get_unique_union(all_retrieval)

            print(f"[Worker {worker_idx}] step4: rerank ({len(candidates)} candidates)", flush=True)
            pairs = [[question, d.page_content] for d in candidates]
            scores = rerank_via_service(worker_idx, request_queue, response_queue, pairs)

            print(f"[Worker {worker_idx}] step5: generate_answer", flush=True)
            scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            docs   = [d for d, _ in scored[:rag_pipeline.rerank_top_k]]
            answer = rag_pipeline._generate_answer(question, docs)

            results.append({
                "_original_idx": original_idx,
                "title":    title,
                "question": question,
                "answer":   answer,
                "evidence": [d.page_content for d in docs],
            })

        except Exception as e:
            print(f"[Worker {worker_idx}] ERROR on '{title}': {e}", flush=True)
            results.append({
                "_original_idx": original_idx,
                "title":    title,
                "question": question,
                "answer":   f"ERROR: {e}",
                "evidence": [],
            })

    result_queue.put(results)