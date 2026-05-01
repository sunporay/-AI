import os
import pickle
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from filelock import FileLock
from langchain_community.vectorstores import FAISS
from config import NUM_GPUS
from embeddings import get_emb_splitter
from retriever import create_bm25_index


def index_worker(args):
    idx, paper, gpu_id, persist_dir = args

    emb, splitter = get_emb_splitter(gpu_id)

    title      = paper.get("title", f"paper_{idx}")
    index_path = os.path.join(
        persist_dir,
        "paper_" + hashlib.md5(title.encode()).hexdigest()[:12],
    )
    bm25_path   = os.path.join(index_path, "bm25.pkl")
    splits_path = os.path.join(index_path, "splits.pkl")
    lock_path   = index_path + ".lock"

    with FileLock(lock_path):
        if (
            os.path.exists(index_path)
            and os.path.exists(bm25_path)
            and os.path.exists(splits_path)
        ):
            return  # 已存在，跳過

        os.makedirs(index_path, exist_ok=True)
        splits = splitter.create_documents(
            [paper["full_text"]], metadatas=[{"title": title}]
        )
        vectorstore = FAISS.from_documents(splits, emb)
        vectorstore.save_local(index_path)
        bm25 = create_bm25_index(splits)
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)
        with open(splits_path, "wb") as f:
            pickle.dump(splits, f)


def index_parallel(papers, persist_dir: str, num_workers: int = NUM_GPUS):
    ctx = multiprocessing.get_context("spawn")
    tasks = [
        (idx, paper, idx % NUM_GPUS, persist_dir)
        for idx, paper in enumerate(papers)
    ]
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
        list(tqdm(ex.map(index_worker, tasks), total=len(tasks), desc="Indexing"))