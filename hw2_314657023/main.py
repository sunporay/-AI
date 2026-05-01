import os
import argparse
import hashlib
import torch
from tqdm import tqdm
from transformers import pipeline as hf_pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from config import NUM_GPUS
from embeddings import QwenEmbeddings, get_emb_splitter
from indexer import index_parallel
from pipeline import PaperRAGPipeline
from reranker import OfficialQwenReranker
from retriever import fusion_retrieval, get_unique_union
from utils import load_json, save_json
from HyDE import HyDE

def cmd_index(args):
    persist_dir = os.path.abspath(args.persist_dir)
    papers = load_json(args.input)

    if args.force:
        import shutil
        for i, p in enumerate(papers):
            folder = "paper_" + hashlib.md5(
                p.get("title", f"paper_{i}").encode()
            ).hexdigest()[:12]
            path = os.path.join(persist_dir, folder)
            if os.path.exists(path):
                shutil.rmtree(path)

    if args.parallel:
        index_parallel(papers, persist_dir=persist_dir, num_workers=args.workers)
    else:
        emb, _ = get_emb_splitter(gpu_id=0)
        pipeline = PaperRAGPipeline(embedding_model=emb, persist_dir=persist_dir)
        for i, p in enumerate(tqdm(papers, desc="Indexing")):
            pipeline._build_index(p["full_text"], p.get("title", f"paper_{i}"))

    print("✅ index done")


def cmd_query(args):
    persist_dir = os.path.abspath(args.persist_dir)
    print(f"[INFO] Single process mode, HuggingFace LLM on cuda:1")

    papers = load_json(args.input)
    emb, _ = get_emb_splitter(gpu_id=0)

    pipe = hf_pipeline(
    "text-generation",
    model="meta-llama/Llama-3.2-3B-Instruct",
    device="cuda:1",
    dtype=torch.float16,
    return_full_text=False,
    max_new_tokens=512,
    temperature=0,
    do_sample=False,
    )
    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

    reranker = OfficialQwenReranker(batch_size=1)
    rag_pipeline = PaperRAGPipeline(
        embedding_model=emb,
        llm=llm,
        alpha=0.5,
        fusion_k=40,
        rerank_top_k=4,
        num_queries=1,
        persist_dir=persist_dir,
    )
    rag_pipeline.mq_expander = HyDE(llm=llm, num_hypothetical=1)
    results = []
    for i, paper in enumerate(tqdm(papers, desc="Querying")):
        title     = paper.get("title", f"paper_{i}")
        question  = paper["question"]
        full_text = paper["full_text"]

        try:
            splits, vectorstore, bm25 = rag_pipeline._build_index(full_text, title)
            queries = rag_pipeline.mq_expander.generate_queries(question)
            all_retrieval = [
                fusion_retrieval(vectorstore, bm25, splits, q,
                    k=rag_pipeline.fusion_k, alpha=rag_pipeline.alpha)
                for q in queries
            ]
            candidates = get_unique_union(all_retrieval)
            pairs = [[question, d.page_content] for d in candidates]
            scores = reranker.predict(pairs)
            scored = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            docs   = [d for d, _ in scored[:rag_pipeline.rerank_top_k]]
            answer = rag_pipeline._generate_answer(question, docs)

            results.append({
                "title":    title,
                "answer":   answer,
                "evidence": [d.page_content for d in docs],
            })
        except Exception as e:
            print(f"ERROR on '{title}': {e}")
            results.append({
                "title":    title,
                "answer":   f"ERROR: {e}",
                "evidence": [],
            })

    save_json(results, args.output)
    print(f"✅ query done → {args.output}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("index")
    p1.add_argument("--input",       default=os.path.join(DATA_DIR, "private_dataset.json"))
    p1.add_argument("--persist-dir", default="./faiss_db")
    p1.add_argument("--force",       action="store_true")
    p1.add_argument("--parallel",    action="store_true")
    p1.add_argument("--workers",     type=int, default=NUM_GPUS)

    p2 = sub.add_parser("query")
    p2.add_argument("--input",         default=os.path.join(DATA_DIR, "private_dataset.json"))
    p2.add_argument("--output",        default=os.path.join(DATA_DIR, "out.json"))
    p2.add_argument("--persist-dir",   default=os.path.join(BASE_DIR, "faiss_db"))
    p2.add_argument("--num-processes", type=int, default=NUM_GPUS)
    p2.add_argument("--base-port",     type=int, default=11434)

    parsed = parser.parse_args()
    if parsed.cmd == "index":
        cmd_index(parsed)
    elif parsed.cmd == "query":
        cmd_query(parsed)