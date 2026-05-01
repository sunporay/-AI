import os
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from config import NUM_GPUS
from embeddings import get_emb_splitter
from indexer import index_parallel
from pipeline import PaperRAGPipeline
from config import ensure_hf_login
from prepare_data import cmd_prepare_data as _prepare
from train import cmd_train as _train  # ✅ 從 train.py 導入
from inference import cmd_infer as _infer  # ✅ 從 inference.py 導入

def cmd_prepare_data(args):
    _prepare(args)

def cmd_train(args):
    _train(args)

def cmd_infer(args):
    _infer(args)

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_papers_from_md(parsered_dir: str) -> list[dict]:
    papers = []
    root = Path(parsered_dir)
    for md_path in sorted(root.rglob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        if not text.strip():
            print(f"[SKIP] empty: {md_path}")
            continue
        rel = md_path.relative_to(root)
        papers.append({
            "title":     md_path.stem,
            "full_text": text,
            "rel_path":  str(rel.with_suffix("")),
        })
    print(f"[INFO] Loaded {len(papers)} papers from {parsered_dir}")
    return papers


# ──────────────────────────────────────────────────────────────────────────────
# index
# ──────────────────────────────────────────────────────────────────────────────

def cmd_index(args):
    persist_dir = os.path.abspath(args.persist_dir)
    papers = load_papers_from_md(args.parsered_dir)

    if not papers:
        print("[ERROR] No papers loaded, check --parsered-dir path.")
        return

    if args.force:
        for p in papers:
            path = os.path.join(persist_dir, p["rel_path"])
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"[FORCE] removed {path}")

    if args.parallel:
        index_parallel(papers, persist_dir=persist_dir, num_workers=args.workers)
    else:
        emb, _ = get_emb_splitter(gpu_id=0)
        pipeline = PaperRAGPipeline(embedding_model=emb, persist_dir=persist_dir)
        for i, p in enumerate(tqdm(papers, desc="Indexing")):
            try:
                pipeline._build_index(
                    p["full_text"],
                    p.get("title", f"paper_{i}"),
                    p.get("rel_path"),
                )
            except Exception as e:
                print(f"[ERROR] {p.get('title', i)}: {e}")

    print("✅ index done")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ensure_hf_login()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PREP_DIR = os.path.join(BASE_DIR, "prepared_data")

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ===== index =====
    p1 = sub.add_parser("index")
    p1.add_argument("--parsered-dir", default="/data2/408_Ray/HW3/parsered")
    p1.add_argument("--persist-dir",  default="./faiss")
    p1.add_argument("--force",        action="store_true")
    p1.add_argument("--parallel",     action="store_true")
    p1.add_argument("--workers",      type=int, default=NUM_GPUS)

    # ===== prepare-data =====
    p0 = sub.add_parser("prepare-data", help="生成 query rewrite + rerank 後的 train/dev/test examples")
    p0.add_argument("--train-csv",
                    default=os.path.join(DATA_DIR, "train.csv"),
                    help="訓練資料 CSV")
    p0.add_argument("--dev-csv",
                    default=os.path.join(DATA_DIR, "dev.csv"),
                    help="驗證資料 CSV")
    p0.add_argument("--test-csv",
                    default=os.path.join(DATA_DIR, "test.csv"),
                    help="測試資料 CSV")
    p0.add_argument("--persist-dir",
                    default=os.path.join(BASE_DIR, "faiss"))
    p0.add_argument("--output-dir",
                    default=PREP_DIR,
                    help="生成後 examples JSON 的輸出資料夾")
    p0.add_argument("--train-examples",
                    default=os.path.join(PREP_DIR, "train_examples.json"))
    p0.add_argument("--dev-examples",
                    default=os.path.join(PREP_DIR, "dev_examples.json"))
    p0.add_argument("--test-examples",
                    default=os.path.join(PREP_DIR, "test_examples.json"))
    p0.add_argument("--include-test", action="store_true",
                    help="連 test examples 一起先生成")
    p0.add_argument("--force", action="store_true",
                    help="若 examples JSON 已存在則刪掉重建")
    p0.add_argument("--rewrite-model",
                    default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                    help="query rewrite 使用的模型")
    p0.add_argument("--rewrite-max-new-tokens", type=int, default=128)
    p0.add_argument("--max-queries", type=int, default=3)
    p0.add_argument("--fusion-k", type=int, default=40)
    p0.add_argument("--alpha", type=float, default=0.5)
    p0.add_argument("--rerank-top-k", type=int, default=3)

    # ===== train =====
    p2 = sub.add_parser("train", help="QLoRA fine-tuning on hallucination detection")
    p2.add_argument("--train-examples",
                    default=os.path.join(PREP_DIR, "train_examples.json"),
                    help="prepare-data 產生的 train examples JSON")
    p2.add_argument("--dev-examples",
                    default=os.path.join(PREP_DIR, "dev_examples.json"),
                    help="prepare-data 產生的 dev examples JSON")
    p2.add_argument("--checkpoint-dir",
                    default=os.path.join(BASE_DIR, "checkpoints"))
    p2.add_argument("--model-name",
                    default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit")
    p2.add_argument("--epochs",     type=int,   default=3)
    p2.add_argument("--batch-size", type=int,   default=1)
    p2.add_argument("--grad-accum", type=int,   default=8)
    p2.add_argument("--lr",         type=float, default=1e-4)
    p2.add_argument("--eval-batch-size", type=int, default=8,
                    help="dev eval 時的 batch size")
    p2.add_argument("--oversample-ratio", type=float, default=5.0,
                    help="少數類別最大 oversampling 倍率 (default: 5.0)")
    p2.add_argument("--no-oversample", dest="oversample", action="store_false",
                    help="停用 minority class oversampling")
    p2.add_argument("--weight-decay", type=float, default=0.01,
                    help="Weight decay for AdamW optimizer (default: 0.01)")
    p2.add_argument("--neftune-noise-alpha", type=float, default=5.0,
                    help="NEFTune noise alpha for embedding noise (default: 5.0, set 0 to disable)")
    p2.add_argument("--no-rslora", dest="use_rslora", action="store_false",
                    help="Disable rsLoRA (rank-stabilized LoRA)")
    p2.add_argument("--max-length", type=int, default=768,
                    help="Maximum sequence length (default: 768)")

    p2.set_defaults(oversample=True, use_rslora=True)

    # ===== infer =====
    p3 = sub.add_parser("infer", help="用微調後的 checkpoint 對 test.csv 做推論，輸出 id/label CSV")
    p3.add_argument("--test-csv",
                    default=os.path.join(DATA_DIR, "test.csv"),
                    help="測試資料 CSV (欄位: id, claim/question, full_text, title)")
    p3.add_argument("--output",
                    default=os.path.join(DATA_DIR, "submission.csv"),
                    help="輸出 CSV 路徑")
    p3.add_argument("--checkpoint",
                    default=os.path.join(BASE_DIR, "checkpoints", "best"),
                    help="微調後的 checkpoint 目錄 (含 adapter_config.json)")
    p3.add_argument("--base-model",
                    default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                    help="LoRA 對應的 base model 名稱")
    p3.add_argument("--persist-dir",
                    default=os.path.join(BASE_DIR, "faiss"),
                    help="FAISS index 根目錄")
    p3.add_argument("--test-examples",
                    default=os.path.join(PREP_DIR, "test_examples.json"),
                    help="prepare-data 產生的 test examples JSON；若不存在會自動生成")
    p3.add_argument("--infer-batch-size", type=int, default=8,
                    help="inference 時的 batch size")
    p3.add_argument("--force-prepare-test", action="store_true",
                    help="忽略既有 test examples，重新生成")
    p3.add_argument("--rewrite-model",
                    default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                    help="自動生成 test examples 時 query rewrite 使用的模型")
    p3.add_argument("--rewrite-max-new-tokens", type=int, default=128)
    p3.add_argument("--max-length", type=int, default=768,
                    help="Maximum sequence length for inference (default: 768)")
    p3.add_argument("--max-queries", type=int, default=3)
    p3.add_argument("--fusion-k", type=int, default=40)
    p3.add_argument("--alpha", type=float, default=0.5)
    p3.add_argument("--rerank-top-k", type=int, default=4)

    parsed = parser.parse_args()
    if parsed.cmd == "index":
        cmd_index(parsed)
    elif parsed.cmd == "prepare-data":
        cmd_prepare_data(parsed)
    elif parsed.cmd == "train":
        cmd_train(parsed)
    elif parsed.cmd == "infer":
        cmd_infer(parsed)