"""inference.py — Complete automated inference pipeline with multi-GPU support.

Full workflow:
1. Parse PDFs to markdown (if needed)
2. Build FAISS index (if needed) - supports multi-GPU parallel processing
3. Prepare test examples (query rewrite + retrieval + rerank)
4. Load fine-tuned model
5. Batch inference
6. Output results

Usage:
    python inference.py                       # Auto everything (single GPU)
    python inference.py --workers 4           # Use 4 GPUs for indexing
    python inference.py --force-parse         # Force re-parse PDFs
    python inference.py --no-parse            # Skip PDF parsing
    python inference.py --force-index         # Force rebuild index
"""

import os
import csv
import re
import sys
import torch
import argparse
import shutil
import multiprocessing
from collections import Counter
from pathlib import Path

from model_utils import load_finetuned
from prepare_data import (
    GPU_CHAT,
    GPU_EMB,
    GPU_RERANK,
    GPU_REWRITE,
    RERANK_BATCH_SIZE,
    RERANK_MAX_LENGTH,
    LABEL2ID,
    ID2LABEL,
    QWEN_CHATML_PATTERN,
    resolve_examples_path,
    prepare_examples_from_csv,
    _load_cache,
)


# ─── PDF parsing functionality ────────────────────────────────────────────────

def parse_test_pdfs(input_dir: str, output_dir: str, force: bool = False):
    """Parse all PDFs in input_dir to markdown files in output_dir."""
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import PdfFormatOption
    from tqdm import tqdm

    input_root = Path(input_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Find all PDFs
    pdf_files = list(input_root.rglob("*.pdf"))
    if not pdf_files:
        print(f"[WARN] No PDF files found in {input_dir}")
        return

    print(f"[INFO] Found {len(pdf_files)} PDF files in {input_dir}")

    # Filter out already parsed files if not forcing
    if not force:
        to_parse = []
        for pdf_path in pdf_files:
            relative_path = pdf_path.relative_to(input_root)
            md_path = output_root / relative_path.with_suffix(".md")
            if not md_path.exists():
                to_parse.append(pdf_path)
            else:
                print(f"[SKIP] Already parsed: {relative_path}")
        pdf_files = to_parse

    if not pdf_files:
        print("[INFO] All PDFs already parsed. Use --force-parse to re-parse.")
        return

    print(f"[INFO] Parsing {len(pdf_files)} PDF files...")

    # Setup docling converter
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Process each PDF
    for pdf_path in tqdm(pdf_files, desc="Parsing PDFs"):
        relative_path = pdf_path.relative_to(input_root)
        md_path = output_root / relative_path.with_suffix(".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = converter.convert(str(pdf_path))
            markdown_text = result.document.export_to_markdown()

            if not markdown_text.strip():
                print(f"[WARN] Empty output: {relative_path}")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)

            print(f"[OK] {relative_path} → {md_path.relative_to(output_root)}")

        except Exception as e:
            print(f"[ERROR] Failed to parse {relative_path}: {e}")

    print(f"✅ PDF parsing complete → {output_root}")


# ─── FAISS index building (multi-GPU support) ─────────────────────────────────

def load_papers_from_md(parsed_dir: str) -> list[dict]:
    """Load all markdown files from parsed directory."""
    papers = []
    root = Path(parsed_dir)
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
    print(f"[INFO] Loaded {len(papers)} papers from {parsed_dir}")
    return papers


def _build_index_worker(gpu_id: int, papers: list[dict], persist_dir: str):
    """Worker function for parallel index building on a specific GPU."""
    # Set CUDA_VISIBLE_DEVICES to use only this GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from embeddings import get_emb_splitter
    from pipeline import PaperRAGPipeline
    from tqdm import tqdm

    print(f"[GPU {gpu_id}] Starting to process {len(papers)} papers")

    try:
        # Load embedding model (will use GPU 0 since CUDA_VISIBLE_DEVICES is set)
        emb, _ = get_emb_splitter(gpu_id=0)
        pipeline = PaperRAGPipeline(embedding_model=emb, persist_dir=persist_dir)

        # Process papers
        for i, p in enumerate(tqdm(papers, desc=f"GPU {gpu_id}", position=gpu_id, leave=False)):
            try:
                pipeline._build_index(
                    p["full_text"],
                    p.get("title", f"paper_{i}"),
                    p.get("rel_path"),
                )
            except Exception as e:
                print(f"[GPU {gpu_id}] ERROR {p.get('title', i)}: {e}")

        del emb, pipeline
        torch.cuda.empty_cache()

        print(f"[GPU {gpu_id}] ✅ Completed {len(papers)} papers")

    except Exception as e:
        print(f"[GPU {gpu_id}] ❌ Worker failed: {e}")
        import traceback
        traceback.print_exc()


def _chunk_list(lst: list, n: int) -> list[list]:
    """Split list into n roughly equal chunks."""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def build_index_parallel(papers: list[dict], persist_dir: str, num_workers: int = 4):
    """Build FAISS index for papers using multiple GPUs in parallel."""
    if not papers:
        print("[ERROR] No papers to index")
        return

    print(f"[INFO] 🚀 Parallel indexing with {num_workers} GPUs")
    print(f"[INFO] Processing {len(papers)} papers")

    # Split papers into chunks
    chunks = _chunk_list(papers, num_workers)

    # Show distribution
    for i, chunk in enumerate(chunks):
        if chunk:
            print(f"[INFO]   GPU {i}: {len(chunk)} papers")

    # Create processes
    processes = []
    for gpu_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = multiprocessing.Process(
            target=_build_index_worker,
            args=(gpu_id, chunk, persist_dir)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print(f"✅ Parallel index building complete → {persist_dir}")


def build_index_for_papers(papers: list[dict], persist_dir: str, force: bool = False, num_workers: int = 1):
    """Build FAISS index for all papers (single or multi-GPU)."""
    from embeddings import get_emb_splitter
    from pipeline import PaperRAGPipeline
    from tqdm import tqdm

    if not papers:
        print("[ERROR] No papers to index")
        return

    persist_dir = os.path.abspath(persist_dir)

    # Check if indexes already exist
    if not force:
        existing = []
        missing = []
        for p in papers:
            index_path = os.path.join(persist_dir, p["rel_path"])
            if os.path.exists(index_path):
                existing.append(p["rel_path"])
            else:
                missing.append(p)

        if existing:
            print(f"[INFO] Found {len(existing)} existing indexes")

        if not missing:
            print("[INFO] All papers already indexed. Use --force-index to rebuild.")
            return

        papers = missing
        print(f"[INFO] Building index for {len(papers)} new papers")
    else:
        # Force rebuild: remove existing indexes
        for p in papers:
            index_path = os.path.join(persist_dir, p["rel_path"])
            if os.path.exists(index_path):
                shutil.rmtree(index_path)
                print(f"[FORCE] removed {index_path}")

    # Choose parallel or single GPU
    if num_workers > 1:
        build_index_parallel(papers, persist_dir, num_workers)
    else:
        # Single GPU mode
        print(f"[INFO] Building index on single GPU {GPU_EMB}")
        emb, _ = get_emb_splitter(gpu_id=GPU_EMB)
        pipeline = PaperRAGPipeline(embedding_model=emb, persist_dir=persist_dir)

        try:
            for i, p in enumerate(tqdm(papers, desc="Building indexes")):
                try:
                    pipeline._build_index(
                        p["full_text"],
                        p.get("title", f"paper_{i}"),
                        p.get("rel_path"),
                    )
                except Exception as e:
                    print(f"[ERROR] {p.get('title', i)}: {e}")
        finally:
            del emb, pipeline
            torch.cuda.empty_cache()

        print(f"✅ Index building complete → {persist_dir}")


# ─── Single-item inference ────────────────────────────────────────────────────

def predict_label_id(model, tokenizer, prompt: str, max_length: int = 768) -> int:
    """Predict label ID for a single prompt."""
    orig_trunc_side = tokenizer.truncation_side
    tokenizer.truncation_side = "left"
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    finally:
        tokenizer.truncation_side = orig_trunc_side

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    raw     = tokenizer.decode(new_ids, skip_special_tokens=True)
    raw     = QWEN_CHATML_PATTERN.sub("", raw).strip()
    return _parse_label(raw)


def _parse_label(raw: str) -> int:
    """Parse generated text into label ID."""
    # First try: extract digit 0-4 using regex
    match = re.search(r'\b([0-4])\b', raw)
    if match:
        return int(match.group(1))

    # Second try: match full label names
    for lbl in sorted(LABEL2ID.keys(), key=len, reverse=True):
        if lbl.lower() in raw.lower():
            return LABEL2ID[lbl]

    # Fallback: default to 4 (Temporal - least common class)
    print(f"[WARN] Cannot parse label from: '{raw}'. Defaulting to 4 (Temporal).")
    return 4


# ─── Batch inference ──────────────────────────────────────────────────────────

def predict_batch(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 8,
    max_length: int = 768,
) -> list[int]:
    """Batched generation with LEFT padding for causal LMs."""
    from tqdm import tqdm

    results = []
    device  = next(model.parameters()).device

    orig_padding_side         = tokenizer.padding_side
    orig_truncation_side      = tokenizer.truncation_side
    tokenizer.padding_side    = "left"
    tokenizer.truncation_side = "left"

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="Batch predict", leave=False):
            batch  = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            for seq in out:
                new_ids = seq[input_len:]
                raw     = tokenizer.decode(new_ids, skip_special_tokens=True)
                raw     = QWEN_CHATML_PATTERN.sub("", raw).strip()
                results.append(_parse_label(raw))
    finally:
        tokenizer.padding_side    = orig_padding_side
        tokenizer.truncation_side = orig_truncation_side

    return results


# ─── cmd_infer ────────────────────────────────────────────────────────────────

def cmd_infer(args):
    """Complete automated inference pipeline."""
    from embeddings import get_emb_splitter
    from reranker import OfficialQwenReranker
    from prepare_data import setup_query_rewriter

    print("\n" + "="*70)
    print("🚀 完整自動化推論流程")
    print("="*70 + "\n")

    # ═══ STEP 1: Parse PDFs ═══
    if getattr(args, "auto_parse_test", False):
        test_pdf_dir = getattr(args, "test_pdf_dir", "/data2/408_Ray/HW3C/paper_evidence/test")
        parsed_output_dir = getattr(args, "parsed_test_dir", "/data2/408_Ray/HW3C/parsered/test")
        force_parse = getattr(args, "force_parse_test", False)

        print("="*70)
        print("[STEP 1/5] 解析 PDF 文件")
        print(f"  輸入:  {test_pdf_dir}")
        print(f"  輸出:  {parsed_output_dir}")
        print(f"  強制:  {force_parse}")
        print("="*70 + "\n")

        parse_test_pdfs(test_pdf_dir, parsed_output_dir, force=force_parse)
        print()

    # ═══ STEP 2: Build FAISS Index ═══
    if getattr(args, "auto_index", True):
        parsed_dir = getattr(args, "parsed_test_dir", "/data2/408_Ray/HW3C/parsered/test")
        persist_dir = os.path.abspath(args.persist_dir)
        force_index = getattr(args, "force_index", False)
        num_workers = getattr(args, "workers", 1)

        print("="*70)
        print("[STEP 2/5] 建立 FAISS 索引")
        print(f"  Markdown: {parsed_dir}")
        print(f"  索引輸出: {persist_dir}")
        print(f"  強制重建: {force_index}")
        print(f"  並行 GPU: {num_workers}")
        print("="*70 + "\n")

        # Load papers from all parsed directories (train, dev, test)
        all_parsed_dirs = []
        base_parsed_dir = os.path.dirname(parsed_dir) if parsed_dir.endswith("/test") else parsed_dir

        # Check for train, dev, test subdirectories
        for subdir in ["train", "dev", "test"]:
            potential_dir = os.path.join(base_parsed_dir, subdir)
            if os.path.exists(potential_dir):
                all_parsed_dirs.append(potential_dir)

        # If no subdirectories found, use the parsed_dir itself
        if not all_parsed_dirs:
            all_parsed_dirs = [parsed_dir]

        all_papers = []
        for pdir in all_parsed_dirs:
            papers = load_papers_from_md(pdir)
            all_papers.extend(papers)

        if all_papers:
            build_index_for_papers(all_papers, persist_dir, force=force_index, num_workers=num_workers)
        else:
            print("[WARN] No papers found to index")
        print()

    # ═══ STEP 3: Prepare paths and directories ═══
    persist_dir = os.path.abspath(args.persist_dir)
    output_path = os.path.abspath(args.output)
    test_cache  = resolve_examples_path(args.test_examples)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(test_cache)  or ".", exist_ok=True)

    # ═══ STEP 4: Prepare test examples ═══
    if getattr(args, "force_prepare_test", False) and os.path.exists(test_cache):
        os.remove(test_cache)
        print(f"[FORCE] removed {test_cache}")

    if not os.path.exists(test_cache):
        print("="*70)
        print("[STEP 3/5] 準備測試資料")
        print(f"  快取: {test_cache}")
        print("="*70 + "\n")

        emb, _ = get_emb_splitter(gpu_id=GPU_EMB)
        query_rewriter = setup_query_rewriter(
            model_name     = args.rewrite_model,
            gpu_id         = GPU_REWRITE,
            max_new_tokens = args.rewrite_max_new_tokens,
        )
        reranker = OfficialQwenReranker(
            gpu_id     = GPU_RERANK,
            batch_size = RERANK_BATCH_SIZE,
            max_length = RERANK_MAX_LENGTH,
        )
        try:
            prepare_examples_from_csv(
                csv_path        = args.test_csv,
                persist_dir     = persist_dir,
                embedding_model = emb,
                reranker        = reranker,
                query_rewriter  = query_rewriter,
                has_label       = False,
                cache_path      = test_cache,
                desc            = "Preparing test",
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
        print()

    # ═══ STEP 5: Load test examples ═══
    test_examples = _load_cache(test_cache)
    print(f"[INFO] 載入 {len(test_examples)} 個測試樣本\n")

    # ═══ STEP 6: Load fine-tuned model ═══
    print("="*70)
    print("[STEP 4/5] 載入微調模型")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Base model: {args.base_model}")
    print(f"  GPU:        {GPU_CHAT}")
    print("="*70 + "\n")

    model, tokenizer = load_finetuned(
        args.base_model,
        args.checkpoint,
        device_map={"": GPU_CHAT},
    )

    # ═══ STEP 7: Run batch inference ═══
    print("="*70)
    print("[STEP 5/5] 執行批次推論")
    print(f"  批次大小: {args.infer_batch_size}")
    print(f"  最大長度: {getattr(args, 'max_length', 768)}")
    print("="*70 + "\n")

    prompts   = [ex["prompt"] for ex in test_examples]
    infer_max_length = getattr(args, "max_length", 768)
    label_ids = predict_batch(
        model, tokenizer, prompts,
        batch_size=args.infer_batch_size,
        max_length=infer_max_length,
    )

    # ═══ STEP 8: Save results ═══
    rows = [{"id": ex["id"], "label": lid} for ex, lid in zip(test_examples, label_ids)]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label"])
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "="*70)
    print("✅ 推論完成！")
    print(f"  輸出: {output_path}")
    print(f"  樣本數: {len(rows)}")
    print("="*70 + "\n")

    # Label distribution
    label_dist = Counter(r["label"] for r in rows)
    print("標籤分布:")
    for lid in sorted(label_dist):
        print(f"   {lid} ({ID2LABEL[lid]:22s}): {label_dist[lid]}")
    print()


# ─── Standalone script functionality ──────────────────────────────────────────

def main():
    """Main entry point when running as a standalone script."""
    multiprocessing.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(
        description="完整自動化推論流程：PDF解析 → 建立索引（多GPU） → 推論 → 輸出結果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 完整自動化流程（單 GPU）
  python inference.py

  # 使用 4 個 GPU 並行建立索引
  python inference.py --workers 4

  # 使用 8 個 GPU 並行建立索引
  python inference.py --workers 8

  # 強制重新解析 PDF 和重建索引（4 GPU）
  python inference.py --force-parse --force-index --workers 4

  # 跳過 PDF 解析（已解析過）
  python inference.py --no-parse --workers 4

  # 跳過索引建立（已建立過）
  python inference.py --no-index

  # 自訂路徑
  python inference.py --checkpoint ./checkpoints/epoch_2 --output ./results.csv
        """
    )

    # PDF parsing options
    parser.add_argument(
        "--auto-parse",
        dest="auto_parse_test",
        action="store_true",
        default=True,
        help="自動解析 test PDF 檔案（預設啟用）"
    )
    parser.add_argument(
        "--no-parse",
        dest="auto_parse_test",
        action="store_false",
        help="跳過 PDF 解析步驟"
    )
    parser.add_argument(
        "--force-parse",
        dest="force_parse_test",
        action="store_true",
        help="強制重新解析所有 test PDF"
    )
    parser.add_argument(
        "--test-pdf-dir",
        default="/data2/408_Ray/HW3/paper_evidence/test",
        help="Test PDF 檔案所在目錄"
    )
    parser.add_argument(
        "--parsed-test-dir",
        default="/data2/408_Ray/HW3C/parsered/test",
        help="解析後的 markdown 輸出目錄"
    )

    # Index building options
    parser.add_argument(
        "--auto-index",
        action="store_true",
        default=True,
        help="自動建立 FAISS 索引（預設啟用）"
    )
    parser.add_argument(
        "--no-index",
        dest="auto_index",
        action="store_false",
        help="跳過索引建立步驟"
    )
    parser.add_argument(
        "--force-index",
        action="store_true",
        help="強制重建所有索引"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="並行建立索引的 GPU 數量（預設：1，推薦：4-8）"
    )

    # Inference options
    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/best",
        help="微調後的 checkpoint 目錄（預設：./checkpoints/best）"
    )
    parser.add_argument(
        "--base-model",
        default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        help="LoRA 對應的 base model 名稱"
    )
    parser.add_argument(
        "--output",
        default="./data/submission.csv",
        help="輸出 CSV 路徑（預設：./data/submission.csv）"
    )
    parser.add_argument(
        "--batch-size",
        dest="infer_batch_size",
        type=int,
        default=8,
        help="推論批次大小（預設：8）"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=768,
        help="最大序列長度（預設：768）"
    )

    # Data preparation options
    parser.add_argument(
        "--test-csv",
        default="./data/test.csv",
        help="測試資料 CSV"
    )
    parser.add_argument(
        "--persist-dir",
        default="./faiss_test",
        help="FAISS index 根目錄（預設：./faiss_test）"
    )
    parser.add_argument(
        "--test-examples",
        default="./prepared_data/test_examples.json",
        help="Test examples JSON 快取路徑"
    )
    parser.add_argument(
        "--force-prepare-test",
        action="store_true",
        help="強制重新生成 test examples"
    )

    # Query rewriting options
    parser.add_argument(
        "--rewrite-model",
        default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        help="Query rewrite 使用的模型"
    )
    parser.add_argument(
        "--rewrite-max-new-tokens",
        type=int,
        default=128,
        help="Query rewrite 最大生成 tokens"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=3,
        help="最大查詢數量"
    )
    parser.add_argument(
        "--fusion-k",
        type=int,
        default=40,
        help="Fusion K 參數"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha 參數"
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=4,
        help="Reranking 後保留的文檔數量"
    )

    args = parser.parse_args()

    # 顯示配置
    print("\n" + "="*70)
    print("⚙️  配置參數")
    print("="*70)
    print(f"Checkpoint:      {args.checkpoint}")
    print(f"Base model:      {args.base_model}")
    print(f"輸出檔案:        {args.output}")
    print(f"批次大小:        {args.infer_batch_size}")
    print(f"最大長度:        {args.max_length}")
    if args.auto_parse_test:
        print(f"解析 PDF:        是 {'(強制)' if args.force_parse_test else ''}")
        print(f"  PDF 來源:      {args.test_pdf_dir}")
        print(f"  Markdown 輸出: {args.parsed_test_dir}")
    else:
        print(f"解析 PDF:        否（跳過）")
    if args.auto_index:
        print(f"建立索引:        是 {'(強制)' if args.force_index else ''}")
        print(f"  索引目錄:      {args.persist_dir}")
        print(f"  並行 GPU:      {args.workers}")
    else:
        print(f"建立索引:        否（跳過）")
    print("="*70)

    # 執行推論
    try:
        cmd_infer(args)
        print("="*70)
        print("🎉 所有步驟完成！")
        print(f"結果已儲存至：{args.output}")
        print("="*70 + "\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  使用者中斷執行")
        sys.exit(1)
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ 執行失敗：{e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()