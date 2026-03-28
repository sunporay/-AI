# LLaMA 1B LoRA Fine-tuned for PathoQA Dataset

This project fine-tunes LLaMA-3.2-1B-Instruct using LoRA for the PathoQA Dataset.

---

## Model Info

- **Base Model:** [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- **Method:** LoRA
- **Task:** Multiple Choice QA
- **Dataset:** PathoQA Dataset
- **Dataset Size:** 9,000 samples (Train: 7,200 / Validation: 1,800)
- **Split:** 80% train / 20% validation
- **Format:** Multiple Choice QA (4 options per question)
- **Language:** English

---

## Environment Setup

```bash
conda create -n llama-lora python=3.10
conda activate llama-lora

pip install -r requirements.txt
```

---

## Framework Versions

| Package      | Version  |
|--------------|----------|
| TRL          | 0.24.0   |
| Transformers | 4.57.1   |
| PyTorch      | 2.10.0   |
| Datasets     | 4.8.2    |
| Tokenizers   | 0.22.2   |

---

## LoRA Configuration

| Parameter      | Value                                                        |
|----------------|--------------------------------------------------------------|
| r (rank)       | 16                                                           |
| alpha          | 32                                                           |
| dropout        | 0.18                                                         |
| target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

---

## Training Details

| Hyperparameter | Value              |
|----------------|--------------------|
| Epochs         | 4                  |
| Batch size     | 8                  |
| Learning rate  | 2e-4               |
| Scheduler      | cosine with warmup |
| Max length     | 512                |
| Optimizer      | paged_adamw_8bit   |

---

## Training Procedure

[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Visualize in Weights & Biases" width="150" height="24"/>](https://wandb.ai/poraysun/GAI%20HW1/runs/4cheb6mx)

---

## Results

| Checkpoint | Accuracy |
|------------|----------|
| Epoch 1    | 69.67%   |
| Epoch 2    | 73.39%   |
| Epoch 3    | 74.56%   |
| Epoch 4    | 74.67%   |