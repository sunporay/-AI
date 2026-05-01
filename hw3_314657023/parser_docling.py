import os
import multiprocessing
from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption


def process_pdfs_on_gpu(gpu_id: int, pdf_files: list, input_root: str, output_root: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = "" #str(gpu_id)

    input_root = Path(input_root)
    output_root = Path(output_root)

    print(f"[GPU {gpu_id}] Initializing model ({len(pdf_files)} files)...")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False               # 啟用 OCR（掃描版 PDF）
    pipeline_options.do_table_structure = True   # 啟用表格結構辨識

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    print(f"[GPU {gpu_id}] Model ready.")

    for i, pdf_path_str in enumerate(pdf_files, 1):
        pdf_path = Path(pdf_path_str)
        relative_path = pdf_path.relative_to(input_root)
        md_path = output_root / relative_path.with_suffix(".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[GPU {gpu_id}] ({i}/{len(pdf_files)}) {relative_path}")
        try:
            result = converter.convert(str(pdf_path))
            markdown_text = result.document.export_to_markdown()

            if not markdown_text.strip():
                print(f"[GPU {gpu_id}]  ⚠️  Empty output: {pdf_path.name}")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown_text)

            print(f"[GPU {gpu_id}]  ✅ Saved -> {md_path}")

        except Exception as e:
            print(f"[GPU {gpu_id}]  ❌ Failed: {pdf_path.name} | {e}")


def chunk_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    NUM_GPUS = 8
    input_root = Path("/data2/408_Ray/HW3/paper_evidence")
    output_root = Path("/data2/408_Ray/HW3/parsered")

    pdf_files = [str(p) for p in input_root.rglob("*.pdf")]
    print(f"Found {len(pdf_files)} PDF files, dispatching to {NUM_GPUS} GPUs.\n")

    chunks = chunk_list(pdf_files, NUM_GPUS)

    processes = []
    for gpu_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        p = multiprocessing.Process(
            target=process_pdfs_on_gpu,
            args=(gpu_id, chunk, str(input_root), str(output_root))
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nAll done.")