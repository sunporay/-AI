"""model_utils.py — Shared model setup and utilities for QLoRA training/inference."""

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# ─── Response template for answer-only loss ───────────────────────────────────
# Response template for Qwen ChatML — marks the start of the assistant turn.
# DataCollatorForCompletionOnlyLM will mask every token BEFORE this template,
# so loss is computed only on the answer tokens.
#
# NOTE: We intentionally drop the trailing '\n' here. Reason:
#   Qwen's tokenizer sometimes merges a lone '\n' with the following character
#   into a single multi-char token (e.g. '\n\n' → token 271, or '\nLabel' → one
#   token). When that happens, the id sequence [151644, 77091, 198] ('\n'=198)
#   never appears in the tokenized stream and the collator masks EVERYTHING.
#
#   '<|im_start|>' (151644) and 'assistant' (77091) are both stable tokens, so
#   matching on just [151644, 77091] is unambiguous and robust.
RESPONSE_TEMPLATE = "<|im_start|>assistant"


# ─── Custom DataCollatorForCompletionOnlyLM ───────────────────────────────────
class DataCollatorForCompletionOnlyLM:
    """Mask prompt tokens so loss is computed on answer tokens only.

    Expects features that already contain "input_ids" (i.e. already tokenized).
    Sequences in a batch can have DIFFERENT lengths — this collator pads them
    to the longest in the batch using the tokenizer's pad_token_id.

    Behaviour:
      1. Pad input_ids (+ attention_mask) to batch-max length on the right.
      2. Copy input_ids → labels, then for each sequence, set everything up to
         and INCLUDING the last occurrence of response_template_ids to -100
         (so loss is only computed on answer tokens).
      3. Set label positions where attention_mask == 0 (padding) to -100.

    Compatible with TRL 1.x which dropped this class from its public API.
    """

    def __init__(self, response_template: list[int], tokenizer):
        self.template_ids = response_template
        self.tokenizer    = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        # ── 1. Pad to batch-max length (right-padding for training) ───────────
        input_ids_list = [
            f["input_ids"].tolist() if hasattr(f["input_ids"], "tolist") else list(f["input_ids"])
            for f in features
        ]
        has_attn = "attention_mask" in features[0]
        attn_list = None
        if has_attn:
            attn_list = [
                f["attention_mask"].tolist() if hasattr(f["attention_mask"], "tolist") else list(f["attention_mask"])
                for f in features
            ]

        max_len = max(len(ids) for ids in input_ids_list)

        padded_ids  = []
        padded_attn = []
        for i, ids in enumerate(input_ids_list):
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [self.pad_token_id] * pad_len)
            if has_attn:
                padded_attn.append(attn_list[i] + [0] * pad_len)
            else:
                padded_attn.append([1] * len(ids) + [0] * pad_len)

        input_ids      = torch.tensor(padded_ids,  dtype=torch.long)
        attention_mask = torch.tensor(padded_attn, dtype=torch.long)

        # ── 2. Build labels: mask prompt tokens ──────────────────────────────
        labels = input_ids.clone()
        tpl    = self.template_ids
        t_len  = len(tpl)

        for i in range(labels.size(0)):
            seq_list   = input_ids[i].tolist()
            mask_until = seq_list.__len__()

            for j in range(len(seq_list) - t_len, -1, -1):
                if seq_list[j : j + t_len] == tpl:
                    mask_until = j + t_len
                    break

            labels[i, :mask_until] = -100

        # ── 3. Mask padding positions in labels ──────────────────────────────
        labels[attention_mask == 0] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


# ─── Model configuration ──────────────────────────────────────────────────────

def _bnb_config() -> BitsAndBytesConfig:
    """BitsAndBytes 4-bit quantization config.

    V100 compatibility: automatically selects float16 for V100 (no bfloat16 support)
    and bfloat16 for A100/H100.
    """
    compute_dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )


def supports_bf16_training() -> bool:
    """Check if GPU supports bfloat16 training (A100/H100 yes, V100 no)."""
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def cast_trainable_params(model, dtype: torch.dtype):
    """Cast trainable parameters to specified dtype."""
    for param in model.parameters():
        if param.requires_grad and param.data.dtype != dtype:
            param.data = param.data.to(dtype)


# ─── Model setup ──────────────────────────────────────────────────────────────

def setup_qlora(
    model_name: str,
    device_map: str | dict = "auto",
    use_rslora: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
):
    """Load model for QLoRA training with configurable LoRA parameters.

    Three usage modes:

    A) Single GPU (set CUDA_VISIBLE_DEVICES to one GPU):
         device_map = {"": 0}
         → model pinned to cuda:0, simple and fast for small models.

    B) Multi-GPU via DDP (recommended when model fits on one card):
         launched via `accelerate launch` / `torchrun`, with LOCAL_RANK env var.
         Each process owns one GPU and sees a FULL copy of the model.
         → device_map must pin to the local rank's GPU, NOT "auto".
         → LoRA adapter gradients are all-reduced across GPUs by Accelerate;
           the 4-bit quantized base weights are frozen so NCCL never sees them.

    C) Pipeline parallel via device_map="auto" (only if model is too big for one card):
         single process, model split across GPUs. NOT recommended for 3B QLoRA.

    This function auto-detects DDP via the LOCAL_RANK env var and overrides
    device_map={"": LOCAL_RANK} so the quantized model lands on the right GPU.

    Args:
        model_name: HuggingFace model name
        device_map: Device mapping strategy
        use_rslora: Enable Rank-Stabilized LoRA (default: True)
        lora_r: LoRA rank (default: 16, options: 8/16/32/64)
        lora_alpha: LoRA alpha scaling (default: 32, typically 1-2x of r)
        lora_dropout: LoRA dropout rate (default: 0.05)
    """
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires CUDA, but no GPU is available.")

    # Auto-detect DDP
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        local_rank = int(local_rank)
        device_map = {"": local_rank}
        print(f"[INFO] DDP mode detected (LOCAL_RANK={local_rank}) → "
              f"device_map={device_map}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token       = tokenizer.eos_token
    tokenizer.padding_side    = "right"
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=_bnb_config(),
        device_map=device_map,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA with configurable parameters
    model = get_peft_model(model, LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        use_rslora=use_rslora,
    ))

    print(f"[INFO] LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"[INFO] rsLoRA: {'enabled' if use_rslora else 'disabled'}")
    model.print_trainable_parameters()
    return model, tokenizer


def load_finetuned(
    base_model_name: str,
    checkpoint_dir: str,
    device_map: str | dict = "auto",
):
    """Load LoRA fine-tuned model for inference.

    ✅ AUTO-DETECTS the correct base model from adapter_config.json

    This function automatically reads the base_model_name_or_path from the
    checkpoint's adapter_config.json to ensure the LoRA adapter is loaded
    on the correct base model it was trained on.

    Common mistake: Training on Qwen2.5-3B-Instruct but loading with
    Qwen2.5-Coder-3B-Instruct causes poor inference results even though
    both are "3B models" — they have different weights!

    Args:
        base_model_name: Fallback base model name (will be overridden by config)
        checkpoint_dir: Path to checkpoint directory with adapter files
        device_map: Device mapping strategy

    Returns:
        (model, tokenizer) tuple ready for inference
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)

    # ✅ Read the CORRECT base model from adapter_config.json
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")

    if not os.path.exists(adapter_config_path):
        raise FileNotFoundError(
            f"adapter_config.json not found in {checkpoint_dir}\n"
            f"This doesn't appear to be a valid PEFT checkpoint."
        )

    with open(adapter_config_path, 'r') as f:
        adapter_config = json.load(f)

    # Get the actual base model used during training
    actual_base_model = adapter_config.get("base_model_name_or_path")

    if not actual_base_model:
        raise ValueError(
            f"adapter_config.json doesn't contain 'base_model_name_or_path'.\n"
            f"Config: {adapter_config}"
        )

    # Warn if user provided a different base model
    if base_model_name != actual_base_model:
        print(f"\n{'='*80}")
        print(f"⚠️  BASE MODEL MISMATCH DETECTED!")
        print(f"{'='*80}")
        print(f"Provided:      {base_model_name}")
        print(f"Checkpoint:    {actual_base_model}")
        print(f"\n✅ Auto-corrected to use: {actual_base_model}")
        print(f"   (the base model this checkpoint was trained on)")
        print(f"{'='*80}\n")

    # Use the correct base model
    base_model_to_load = actual_base_model

    print(f"[INFO] Loading base model: {base_model_to_load}")
    print(f"[INFO] Loading adapter from: {checkpoint_dir}")

    # Load tokenizer from checkpoint (has correct special tokens)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"  # ✅ CRITICAL: left-padding for batched causal LM generation
    tokenizer.truncation_side = "left"

    # Load base model with quantization
    base = AutoModelForCausalLM.from_pretrained(
        base_model_to_load,
        quantization_config=_bnb_config(),
        device_map=device_map,
        trust_remote_code=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model.eval()

    print(f"[INFO] ✅ Successfully loaded fine-tuned model")
    print(f"[INFO]    Base model: {base_model_to_load}")
    print(f"[INFO]    Checkpoint: {checkpoint_dir}")
    print(f"[INFO]    Device map: {device_map}")

    return model, tokenizer