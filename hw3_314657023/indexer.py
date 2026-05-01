import os
import pickle
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from filelock import FileLock
from langchain_community.vectorstores import FAISS
from config import NUM_GPUS
from embeddings import get_emb_splitter
from retriever import create_bm25_index


def index_worker(args):
    """处理单个 paper 的索引构建"""
    idx, paper, gpu_id, persist_dir = args

    emb, splitter = get_emb_splitter(gpu_id)

    title      = paper.get("title", f"paper_{idx}")
    rel_path   = paper.get("rel_path", title)          # 例如 train/paper_4
    index_path = os.path.join(persist_dir, rel_path)   # faiss/train/paper_4
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

        full_text = paper.get("full_text", "")
        if not full_text.strip():
            print(f"[SKIP] empty full_text: {title}")
            return

        try:
            os.makedirs(index_path, exist_ok=True)
            splits = splitter.create_documents(
                [full_text], metadatas=[{"title": title}]
            )
            vectorstore = FAISS.from_documents(splits, emb)
            vectorstore.save_local(index_path)
            bm25 = create_bm25_index(splits)
            with open(bm25_path, "wb") as f:
                pickle.dump(bm25, f)
            with open(splits_path, "wb") as f:
                pickle.dump(splits, f)
        except Exception as e:
            print(f"[ERROR] {title}: {e}")


def worker_process_gpu(args):
    """单个 worker 处理分配到特定 GPU 的所有任务（确保一个 GPU 只有一个进程）"""
    gpu_id, gpu_tasks, persist_dir = args

    print(f"[Worker GPU {gpu_id}] Processing {len(gpu_tasks)} papers")

    for idx, paper in tqdm(gpu_tasks, desc=f"GPU {gpu_id}", position=gpu_id):
        # 调用原有的 index_worker 逻辑
        index_worker((idx, paper, gpu_id, persist_dir))


def index_parallel(papers, persist_dir: str, num_workers: int = NUM_GPUS):
    """并行索引构建，每个 GPU 绑定一个 worker 进程"""
    ctx = multiprocessing.get_context("spawn")

    # 按 GPU 分组任务（round-robin 分配）
    gpu_tasks = [[] for _ in range(NUM_GPUS)]
    for idx, paper in enumerate(papers):
        gpu_id = idx % NUM_GPUS
        gpu_tasks[gpu_id].append((idx, paper))

    # 准备每个 worker 的参数：(gpu_id, 该 GPU 的任务列表, persist_dir)
    worker_args = [
        (gpu_id, tasks, persist_dir)
        for gpu_id, tasks in enumerate(gpu_tasks[:num_workers])
        if tasks  # 只处理有任务的 GPU
    ]

    print(f"[INFO] Starting {len(worker_args)} workers on GPUs 0-{len(worker_args)-1}")
    for gpu_id, tasks, _ in worker_args:
        print(f"  GPU {gpu_id}: {len(tasks)} papers")

    # 每个 worker 绑定一个 GPU，按顺序处理该 GPU 的所有任务
    with ProcessPoolExecutor(max_workers=len(worker_args), mp_context=ctx) as ex:
        list(ex.map(worker_process_gpu, worker_args))

    print("✅ Parallel indexing complete")