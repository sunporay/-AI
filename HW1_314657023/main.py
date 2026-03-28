
import torch.distributed as dist
import os
import re
import glob
import torch
import wandb
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainerCallback,
)
from peft import LoraConfig, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig
import shutil

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
IS_MAIN = LOCAL_RANK == 0          # 只有 rank 0 才做 WandB / 存檔

load_dotenv("api_key.env")
wandb_key = os.getenv("WANDB")
hf_token  = os.getenv("HF")
login(token=hf_token)

MODEL_NAME       = "meta-llama/Llama-3.2-1B-Instruct"
seed             = 42
BATCH_SIZE       = 8
EPOCHS           = 4
LR               = 2e-4
LORA_RANK        = 16
LORA_ALPHA       = 32
MAX_TOKEN_LENGTH = 512
drop_out         = 0.18

train_csv_path = r"/data2/408_Ray/HW1/hw-1-question-answering/dataset.csv"
test_csv_path  = r"/data2/408_Ray/HW1/hw-1-question-answering/benchmark.csv"

run_name = f"HW1{LORA_RANK}-a{LORA_ALPHA}_bs{BATCH_SIZE}_lr{LR}_ep{EPOCHS}"


if IS_MAIN:
    wandb.login(key=wandb_key)
    wandb.init(project="GAI HW1", name=run_name)
else:
    os.environ["WANDB_DISABLED"] = "true"   # 非 rank 0 完全關閉 WandB


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,

)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


ds = load_dataset("csv", data_files=train_csv_path)
DS = ds["train"].train_test_split(test_size=0.2, shuffle=True, seed=seed)

def format_prompt(example):
    ans_idx = int(example["ans"])
    letters = ["A", "B", "C", "D"]
    options_text = [
        str(example["opa"]).strip(), str(example["opb"]).strip(),
        str(example["opc"]).strip(), str(example["opd"]).strip(),
    ]
    user_content = (
        f"Below is a multiple choice question. "
        f"Answer with only a single letter A, B, C, or D.\n\n"
        f"Question: {example['question']}\n"
        f"A. {options_text[0]}\nB. {options_text[1]}\n"
        f"C. {options_text[2]}\nD. {options_text[3]}"
    )
    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": f"Answer: {letters[ans_idx]}"},
        ]
    }

formatted_dataset = DS.map(format_prompt)

def preprocess_and_tokenize(example):
    messages  = example["messages"]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full      = tokenizer(full_text, truncation=True, max_length=MAX_TOKEN_LENGTH)
    input_ids = full["input_ids"]

    user_text = tokenizer.apply_chat_template(
        [messages[0]], tokenize=False, add_generation_prompt=True
    )
    user_ids  = tokenizer(user_text)["input_ids"]

    labels = [-100] * len(input_ids)
    for i in range(len(user_ids), len(input_ids)):
        labels[i] = input_ids[i]

    full["labels"] = labels
    return full

tokenized_dataset = formatted_dataset.map(
    preprocess_and_tokenize,
    remove_columns=formatted_dataset["train"].column_names,
)

class EpochCheckpointRenameCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        # 找到剛存的 checkpoint（step 編號）
        if args.local_rank not in (-1, 0):   # ← 只有 rank 0 才做
            return control
        step_ckpt = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        epoch_ckpt = os.path.join(args.output_dir, f"checkpoint-{int(state.epoch)}")
        if os.path.exists(step_ckpt):
            shutil.move(step_ckpt, epoch_ckpt)
        return control

lora_cfg = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=drop_out,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)


args = SFTConfig(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    num_train_epochs=EPOCHS,
    fp16=True,
    output_dir="save_models",  #####
    max_length=MAX_TOKEN_LENGTH,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine_with_min_lr",
    lr_scheduler_kwargs={"min_lr": 1e-6, "num_cycles": 0.5},
    warmup_ratio=0.1,
    learning_rate=float(LR),
    report_to="wandb" if IS_MAIN else "none",   # ← 只有 rank 0 回報
    run_name=run_name,
    logging_steps=1,
    eval_strategy="epoch",
    save_strategy="epoch",
    assistant_only_loss=False,
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=args,
    callbacks=[EpochCheckpointRenameCallback()],
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    peft_config=lora_cfg,
)

trainer.train()
if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()   #
torch.cuda.empty_cache()

if IS_MAIN:

    letter_token_ids = torch.tensor([
        tokenizer.encode(" " + l, add_special_tokens=False)[0]
        for l in ["A", "B", "C", "D"]
    ])
    print(f"ABCD token ids: {letter_token_ids}")

    def predict_one(example, model):
        options_text = [
            str(example["opa"]).strip(), str(example["opb"]).strip(),
            str(example["opc"]).strip(), str(example["opd"]).strip(),
        ]
        user_content = (
            f"Below is a multiple choice question. "
            f"Answer with only a single letter A, B, C, or D.\n\n"
            f"Question: {example['question']}\n"
            f"A. {options_text[0]}\nB. {options_text[1]}\n"
            f"C. {options_text[2]}\nD. {options_text[3]}"
        )
        messages = [{"role": "user", "content": user_content}]
        prompt   = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        scores = []
        for letter in ["A", "B", "C", "D"]:
            full_text = prompt + f"Answer: {letter}"
            inputs    = tokenizer(full_text, return_tensors="pt").to("cuda")
            input_ids = inputs["input_ids"]
            with torch.no_grad():
                outputs = model(**inputs)
            logits   = outputs.logits[0, :-1, :]
            labels   = input_ids[0, 1:]
            log_probs = F.log_softmax(logits, dim=-1)
            score     = log_probs[range(len(labels)), labels].sum().item()
            scores.append(score)

        return scores.index(max(scores))

    checkpoints = sorted(
        glob.glob(f"save_models/checkpoint-*"),
        key=lambda x: int(x.split("-")[-1]),
    )
    print(f"找到 checkpoints: {checkpoints}")

    results_table = wandb.Table(columns=["checkpoint", "accuracy", "correct", "total"])

    for ckpt_path in checkpoints:
        print(f"\n 載入 {ckpt_path}")
        base      = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map={"": "cuda:0"}
        )
        ckpt_model = PeftModel.from_pretrained(base, ckpt_path,device_map={"": "cuda:0"})
        ckpt_model.eval()

        correct = 0
        total   = len(DS["test"])
        for example in tqdm(DS["test"], desc=f"Evaluating {ckpt_path}"):
            if predict_one(example, ckpt_model) == int(example["ans"]):
                correct += 1

        real_acc = correct / total
        print(f" {ckpt_path} Accuracy: {real_acc:.4f} ({correct}/{total})")
        results_table.add_data(ckpt_path, real_acc, correct, total)

        del ckpt_model, base
        torch.cuda.empty_cache()


    wandb.log({
        "checkpoint_accuracy": wandb.plot.bar(
            results_table, "checkpoint", "accuracy", title="Accuracy by Checkpoint"
        )
    })
    wandb.finish()

    lora_model = trainer.model
    lora_model.eval()
    test_ds = load_dataset("csv", data_files=test_csv_path)["train"]

    results = []
    for example in tqdm(test_ds):
        results.append({
            "question_id": example["question_id"],
            "pred": predict_one(example, lora_model),
        })

    pd.DataFrame(results).to_csv("submission.csv", index=False)
    print("Done！共", len(results), "筆")