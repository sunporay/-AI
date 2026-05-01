"""train.py — QLoRA fine-tuning with answer-only loss and class balancing."""

import os
import gc
import math
import random
import inspect
import torch
from collections import Counter
from datasets import Dataset

from sklearn.metrics import f1_score, classification_report

from transformers import (
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from trl import SFTConfig, SFTTrainer

from model_utils import (
    RESPONSE_TEMPLATE,
    DataCollatorForCompletionOnlyLM,
    supports_bf16_training,
    cast_trainable_params,
    setup_qlora,
)

from prepare_data import (
    LABEL2ID,
    ID2LABEL,
    resolve_examples_path,
    _load_cache,
)


# ─── Class-imbalance helpers ──────────────────────────────────────────────────

def compute_class_weights(examples: list[dict]) -> dict[str, float]:
    """Inverse-frequency weights, normalised so the mean weight = 1."""
    counts    = Counter(ex["label"] for ex in examples)
    n_total   = len(examples)
    n_classes = len(counts)
    weights   = {lbl: n_total / (n_classes * cnt) for lbl, cnt in counts.items()}
    return weights


def oversample_to_balance(
    examples: list[dict],
    max_ratio: float = 5.0,
    seed: int = 42,
) -> list[dict]:
    """Oversample minority classes so no class is more than *max_ratio* times
    rarer than the majority class.

    Args:
        examples:  list of dicts, each must have a "label" key.
        max_ratio: cap on majority/minority ratio after oversampling.
                   e.g. 5.0 means minority classes are repeated at most 5×.
        seed:      random seed for reproducibility.

    Returns:
        New list (original + duplicated minority examples), shuffled.
    """
    rng       = random.Random(seed)
    counts    = Counter(ex["label"] for ex in examples)
    max_count = max(counts.values())
    target_count = {
        lbl: min(max_count, math.ceil(cnt * max_ratio))
        for lbl, cnt in counts.items()
    }

    by_label: dict[str, list[dict]] = {}
    for ex in examples:
        by_label.setdefault(ex["label"], []).append(ex)

    balanced = []
    for lbl, exs in by_label.items():
        need = target_count[lbl]
        if need <= len(exs):
            balanced.extend(exs[:need])
        else:
            full_copies = need // len(exs)
            remainder   = need %  len(exs)
            balanced.extend(exs * full_copies)
            balanced.extend(rng.sample(exs, remainder))

    rng.shuffle(balanced)

    # Report
    orig_dist = {lbl: counts[lbl] for lbl in sorted(counts)}
    new_dist  = Counter(ex["label"] for ex in balanced)
    print("[Oversample] Class distribution before → after oversampling:")
    for lbl in sorted(orig_dist):
        print(f"   {lbl:25s}  {orig_dist[lbl]:5d}  →  {new_dist[lbl]:5d}")
    print(f"   Total: {len(examples)} → {len(balanced)}")

    return balanced


# ─── Training callbacks ───────────────────────────────────────────────────────

class EvalAndSaveCallback(TrainerCallback):
    """Evaluate on dev set and checkpoint at the end of every epoch.

    Reports:
      • Overall accuracy
      • Macro F1  (good for imbalanced datasets)
      • Weighted F1
      • Per-class F1 via sklearn classification_report

    Only runs on rank 0 in DDP mode to avoid duplicate evaluation.
    """

    def __init__(
        self,
        dev_examples: list[dict],
        tokenizer,
        checkpoint_dir: str,
        eval_batch_size: int = 8,
        max_length: int = 768,
    ):
        self.dev_examples    = dev_examples
        self.tokenizer       = tokenizer
        self.checkpoint_dir  = checkpoint_dir
        self.eval_batch_size = eval_batch_size
        self.max_length      = max_length
        self.best_f1         = -1.0
        self.best_epoch      = -1

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        # ✅ Only evaluate on rank 0 in DDP mode
        if not state.is_local_process_zero:
            return

        # Import here to avoid circular dependency
        from inference import predict_batch

        epoch = int(state.epoch)
        print(f"\n{'='*60}")
        print(f"[Eval] Epoch {epoch} — running dev evaluation …")

        # ✅ Extract original model from DDP wrapper if needed
        eval_model = model
        if hasattr(model, 'module'):  # DDP wrapper
            eval_model = model.module

        eval_model.eval()
        prompts = [ex["prompt"] for ex in self.dev_examples]

        # ✅ Use torch.no_grad() to save memory during evaluation
        with torch.no_grad():
            pred_ids = predict_batch(
                eval_model,
                self.tokenizer,
                prompts,
                batch_size=self.eval_batch_size,
                max_length=self.max_length,
            )

        eval_model.train()

        # Metrics
        true_labels = [ex["label"] for ex in self.dev_examples]
        pred_labels = [ID2LABEL[pid] for pid in pred_ids]
        label_names = [ID2LABEL[i] for i in range(len(ID2LABEL))]

        correct     = sum(p == t for p, t in zip(pred_labels, true_labels))
        total       = len(self.dev_examples)
        accuracy    = correct / total if total > 0 else 0.0
        macro_f1    = f1_score(true_labels, pred_labels, average="macro",    labels=label_names, zero_division=0)
        weighted_f1 = f1_score(true_labels, pred_labels, average="weighted", labels=label_names, zero_division=0)

        print(f"[Eval] Epoch {epoch}")
        print(f"       Accuracy    : {accuracy:.4f}  ({correct}/{total})")
        print(f"       Macro F1    : {macro_f1:.4f}  ← used for best-checkpoint selection")
        print(f"       Weighted F1 : {weighted_f1:.4f}")
        print(f"\n[Eval] Per-class report:")
        print(classification_report(
            true_labels, pred_labels,
            labels=label_names,
            target_names=label_names,
            zero_division=0,
        ))

        # ✅ Save using original model (not DDP wrapper)
        save_model = eval_model

        # Save checkpoint for this epoch
        ckpt_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}")
        os.makedirs(ckpt_path, exist_ok=True)
        save_model.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        print(f"[Checkpoint] epoch_{epoch} → {ckpt_path}")

        # Update best checkpoint (by macro-F1)
        if macro_f1 > self.best_f1:
            self.best_f1    = macro_f1
            self.best_epoch = epoch
            best_path = os.path.join(self.checkpoint_dir, "best")
            os.makedirs(best_path, exist_ok=True)
            save_model.save_pretrained(best_path)
            self.tokenizer.save_pretrained(best_path)
            print(
                f"[Checkpoint] ★ best (epoch {epoch}, "
                f"macro_f1={macro_f1:.4f}) → {best_path}"
            )

        print(f"{'='*60}\n")


class TrainLossCallback(TrainerCallback):
    """Log training loss every few steps."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if not state.is_local_process_zero or not logs:
            return
        if "loss" not in logs:
            return

        max_steps = state.max_steps if state.max_steps and state.max_steps > 0 else "?"
        loss = logs["loss"]
        lr   = logs.get("learning_rate")
        if lr is None:
            print(f"[Train] step {state.global_step}/{max_steps} - loss: {loss:.6f}")
        else:
            print(
                f"[Train] step {state.global_step}/{max_steps} - "
                f"loss: {loss:.6f} - lr: {lr:.2e}"
            )


# ─── cmd_train ────────────────────────────────────────────────────────────────

def cmd_train(args):
    """Single-process or multi-GPU QLoRA fine-tuning.

    Key improvements:
      1. Answer-only loss via DataCollatorForCompletionOnlyLM
      2. Macro-F1 tracked per epoch (better for imbalanced data)
      3. Minority-class oversampling controlled by --oversample-ratio
      4. rsLoRA enabled by default
      5. NEFTune noise for better generalization
      6. Weight decay for regularization
      7. DDP support with proper memory management
      8. Gradient checkpointing in single-GPU mode (saves ~30% memory)
    """
    # ✅ Memory cleanup at start
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set memory allocation strategy
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    train_cache    = resolve_examples_path(args.train_examples)
    dev_cache      = resolve_examples_path(args.dev_examples)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.exists(train_cache):
        raise FileNotFoundError(
            f"train examples not found: {train_cache}\n"
            f"Run `python main.py prepare-data` first."
        )
    if not os.path.exists(dev_cache):
        raise FileNotFoundError(
            f"dev examples not found: {dev_cache}\n"
            f"Run `python main.py prepare-data` first."
        )

    train_examples = _load_cache(train_cache)
    dev_examples   = _load_cache(dev_cache)
    print(f"[INFO] train: {len(train_examples)} examples, dev: {len(dev_examples)} examples")

    # ✅ Detect DDP mode
    is_ddp = os.environ.get("LOCAL_RANK") is not None
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Class distribution
    train_dist = Counter(ex["label"] for ex in train_examples)
    if local_rank == 0:  # Only print once in DDP mode
        print("[INFO] Training set class distribution:")
        for lbl in sorted(train_dist):
            pct = 100 * train_dist[lbl] / len(train_examples)
            print(f"   {lbl:25s}: {train_dist[lbl]:5d}  ({pct:.1f}%)")

    # Oversample minority classes
    oversample_ratio = getattr(args, "oversample_ratio", 5.0)
    do_oversample    = getattr(args, "oversample", True)

    if do_oversample:
        if local_rank == 0:
            print(f"[INFO] Oversampling minority classes (max_ratio={oversample_ratio}) …")
        train_examples = oversample_to_balance(
            train_examples,
            max_ratio=oversample_ratio,
            seed=42,
        )
    else:
        if local_rank == 0:
            print("[INFO] Oversampling disabled (--no-oversample).")

    # ✅ Determine device map (FIXED for DDP)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs found.")
    elif is_ddp:
        # ✅ DDP mode: always use cuda:0 (torchrun maps it to correct physical GPU)
        device_map = {"": 0}
        if local_rank == 0:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
            print(f"[INFO] DDP mode — each process uses cuda:0 (auto-mapped to physical GPU)")
            print(f"[INFO] Visible GPUs: {visible}, Total processes: {num_gpus}")
    elif num_gpus == 1:
        device_map = {"": 0}
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        print(f"[INFO] Single GPU mode — training on cuda:0 (physical GPU {visible})")
    else:
        device_map = "auto"
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", f"0–{num_gpus - 1}")
        print(f"[INFO] Multi-GPU mode — {num_gpus} GPUs visible ({visible}), device_map='auto'")

    # ✅ Load model with configurable LoRA parameters
    if local_rank == 0:
        print(f"[INFO] Loading model: {args.model_name}")

    use_rslora = getattr(args, "use_rslora", True)
    lora_r = getattr(args, "lora_r", 16)
    lora_alpha = getattr(args, "lora_alpha", 32)
    lora_dropout = getattr(args, "lora_dropout", 0.05)

    model, tokenizer = setup_qlora(
        args.model_name,
        device_map=device_map,
        use_rslora=use_rslora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    # ✅ Gradient checkpointing: enable in single-GPU mode, disable in DDP mode
    use_grad_ckpt = False
    if not is_ddp:
        # Single-GPU or pipeline parallel mode: enable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            use_grad_ckpt = True
            if local_rank == 0:
                print("[INFO] Gradient checkpointing: ENABLED (saves ~30% memory, slightly slower)")
    else:
        # DDP mode: must disable gradient checkpointing
        if hasattr(model, 'gradient_checkpointing_disable'):
            model.gradient_checkpointing_disable()
        if local_rank == 0:
            print("[INFO] Gradient checkpointing: DISABLED (incompatible with DDP)")

    # Answer-only loss collator
    response_template_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer,
    )
    if local_rank == 0:
        print(f"[INFO] DataCollatorForCompletionOnlyLM ready, "
              f"response_template ids = {response_template_ids} "
              f"({tokenizer.decode(response_template_ids)!r})")

    # Build HF Dataset
    hf_train = Dataset.from_list([
        {"text": ex["text"], "length": len(ex["text"])}
        for ex in train_examples
    ])

    # ✅ Sanity check (only on rank 0)
    if local_rank == 0:
        _sample_texts = [hf_train[i]["text"] for i in range(min(2, len(hf_train)))]
        sample_text   = _sample_texts[0]

        print(f"\n[DEBUG] sample text length = {len(sample_text)} chars")
        print(f"[DEBUG] last 300 chars of sample[0]:")
        print(f"        {sample_text[-300:]!r}")

        if RESPONSE_TEMPLATE in sample_text:
            idx = sample_text.rfind(RESPONSE_TEMPLATE)
            print(f"[DEBUG] RESPONSE_TEMPLATE found at char {idx} (of {len(sample_text)})")
            print(f"[DEBUG] 80 chars from template position: {sample_text[idx:idx+80]!r}")
        else:
            print(f"[DEBUG] ❌ RESPONSE_TEMPLATE {RESPONSE_TEMPLATE!r} NOT in text.")

        _sanity_max_length = getattr(args, "max_length", 768)
        _tokenized = tokenizer(
            _sample_texts,
            truncation=True,
            max_length=_sanity_max_length,
            padding=True,
            return_tensors="pt",
        )

        first_ids = _tokenized["input_ids"][0].tolist()
        t_len     = len(response_template_ids)
        hit = None
        for j in range(len(first_ids) - t_len + 1):
            if first_ids[j : j + t_len] == response_template_ids:
                hit = j
                break

        if hit is None:
            print(f"[DEBUG] ❌ template ids {response_template_ids} NOT found in token stream.")
        else:
            print(f"[DEBUG] ✓ template ids found at token position {hit}")

        _sample_features = [
            {
                "input_ids":      _tokenized["input_ids"][i],
                "attention_mask": _tokenized["attention_mask"][i],
            }
            for i in range(len(_sample_texts))
        ]
        sample_batch = data_collator(_sample_features)
        labels       = sample_batch["labels"][0]
        n_masked     = (labels == -100).sum().item()
        n_visible    = (labels != -100).sum().item()
        print(f"[Sanity] labels: {n_masked} masked (prompt), {n_visible} visible (answer)")

        if n_visible == 0:
            raise RuntimeError(
                "DataCollatorForCompletionOnlyLM masked ALL tokens!\n"
                f"  response_template    : {RESPONSE_TEMPLATE!r}\n"
                f"  response_template_ids: {response_template_ids}\n"
                f"  decoded              : {tokenizer.decode(response_template_ids)!r}\n"
            )

        visible_ids = _tokenized["input_ids"][0][labels != -100].tolist()
        print(f"[Sanity] visible token preview: {tokenizer.decode(visible_ids)!r}")

    use_bf16 = supports_bf16_training()
    if local_rank == 0:
        print(f"[INFO] Mixed precision: {'bf16' if use_bf16 else 'fp16 + fp32 LoRA adapters'}")

    max_length = getattr(args, "max_length", 512)  # Default to 512 for memory safety
    if local_rank == 0:
        print(f"[INFO] max_length = {max_length}")

    # ✅ NEFTune and weight_decay parameters
    neftune_noise_alpha = getattr(args, "neftune_noise_alpha", 5.0)
    weight_decay = getattr(args, "weight_decay", 0.01)

    if local_rank == 0:
        print(f"[INFO] NEFTune noise alpha: {neftune_noise_alpha}")
        print(f"[INFO] Weight decay: {weight_decay}")

    # Training arguments
    trainer_supports_processing_class = (
        "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
    )
    trainer_args_cls = SFTConfig if trainer_supports_processing_class else TrainingArguments

    training_args = trainer_args_cls(
        output_dir                    = checkpoint_dir,
        num_train_epochs              = args.epochs,
        per_device_train_batch_size   = args.batch_size,
        gradient_accumulation_steps   = args.grad_accum,
        learning_rate                 = args.lr,
        weight_decay                  = weight_decay,
        max_grad_norm                 = 1.0,  # ✅ 防止梯度爆炸
        fp16                          = not use_bf16,
        bf16                          = use_bf16,
        logging_strategy              = "steps",
        logging_steps                 = 10,
        logging_first_step            = True,
        save_strategy                 = "no",
        warmup_ratio                  = 0.05,
        lr_scheduler_type             = "cosine",
        report_to                     = "none",
        remove_unused_columns         = True,
        dataloader_pin_memory         = False,
        gradient_checkpointing        = use_grad_ckpt,  # ✅ 動態設置
        gradient_checkpointing_kwargs = {"use_reentrant": False} if use_grad_ckpt else None,  # ✅ 動態設置
        optim                         = "paged_adamw_8bit",
        dataloader_num_workers        = 0,
        group_by_length               = False,
        neftune_noise_alpha           = neftune_noise_alpha,
        # ✅ DDP optimizations
        ddp_find_unused_parameters    = False,
        ddp_broadcast_buffers         = False,
        **(
            {"dataset_text_field": "text", "max_length": max_length}
            if trainer_supports_processing_class else {}
        ),
    )

    # Trainer
    trainer_kwargs = dict(
        model         = model,
        args          = training_args,
        train_dataset = hf_train,
        data_collator = data_collator,
        callbacks     = [
            TrainLossCallback(),
            EvalAndSaveCallback(
                dev_examples,
                tokenizer,
                checkpoint_dir,
                eval_batch_size=getattr(args, "eval_batch_size", 2),  # ✅ 4 → 2 省記憶體
                max_length=max_length,
            ),
        ],
    )
    if trainer_supports_processing_class:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"]          = tokenizer
        trainer_kwargs["max_seq_length"]     = max_length
        trainer_kwargs["dataset_text_field"] = "text"

    trainer = SFTTrainer(**trainer_kwargs)
    if not use_bf16:
        cast_trainable_params(trainer.model, torch.float32)

    if local_rank == 0:
        print("[INFO] Starting QLoRA fine-tuning with rsLoRA + NEFTune …")
    trainer.train()

    # ✅ Only save final checkpoint on rank 0
    if local_rank == 0:
        final_path = os.path.join(checkpoint_dir, "final")
        os.makedirs(final_path, exist_ok=True)

        # Extract model from DDP wrapper if needed
        save_model = model
        if hasattr(model, 'module'):
            save_model = model.module

        save_model.save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"\n✅ Training complete → {final_path}")