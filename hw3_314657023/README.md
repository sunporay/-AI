#  PaperRAG: Hallucination Detection Pipeline

##  Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for **hallucination detection in academic papers**, including:

*  Document parsing (PDF → Markdown)
*  FAISS indexing
*  Query rewrite
*  Retrieval + reranking
*  QLoRA fine-tuning
*  Inference (prediction)

---

## 🗂️ Project Structure

```bash
.
├── main.py              # CLI entry point
├── train.py             # QLoRA training
├── inference.py         # Model inference
├── prepare_data.py      # Data preparation (RAG pipeline)
│
├── pipeline.py          # Core RAG pipeline
├── retriever.py         # Retrieval logic
├── reranker.py          # Reranking model
├── query_rewrite.py     # Query rewriting
├── query_worker.py      # Multi-query handling
│
├── embeddings.py        # Embedding model
├── indexer.py           # FAISS indexing
├── model_utils.py       # Model loading helpers
├── parser_docling.py    # PDF parsing (if used)
├── config.py            # Global configs
│
├── data/                # Raw CSV datasets
├── checkpoints/         # Trained LoRA adapters
│
├── requirements.txt
└── README.md
```

---

##  Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### HuggingFace login

```bash
huggingface-cli login
```

---

##  Pipeline Usage

### 1️ Build Index

```bash
python main.py index
```

---

### 2️ Prepare Data (RAG)

```bash
python main.py prepare-data
```

 This step performs:

* Query rewrite
* Retrieval
* Reranking
* Training example generation

---

### 3️ Train (QLoRA)

```bash
python main.py train
```

 Output:

```
checkpoints/
 └── best/
     ├── adapter_model.safetensors
     └── adapter_config.json
```

 This is **LoRA adapter (NOT full model)**

---

### 4️ Inference

```bash
python inference.py
```

---

##  Model Architecture

```text
Input (claim)
   ↓
Query Rewrite
   ↓
Retriever (FAISS)
   ↓
Reranker
   ↓
Context + Claim
   ↓
QLoRA Model
   ↓
Prediction (0~4)
```

---

##  Output Format

```csv
id,label
1,0
2,3
3,1
```

---

##  Tips

### Memory optimization

* Reduce `--max-length`
* Reduce `--batch-size`
* Set `--neftune-noise-alpha 0`

### Performance

* Use `--parallel` for indexing
* Increase `--infer-batch-size`

---

##  Full Example

```bash
python main.py index --parallel
python main.py prepare-data --include-test
python main.py train
python main.py infer
```

---

