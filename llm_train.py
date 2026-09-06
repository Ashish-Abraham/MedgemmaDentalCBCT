"""
train_medgemma_lora.py

QLoRA fine-tuning of google/medgemma-27b-it for the MMDental multimodal
clinical-record generation task (MICCAI STSR 2026, Task 3).
"""

import os
import json
import argparse
import inspect
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


# --------------------------------------------------------------------------- #
# CLI args
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="google/medgemma-27b-it")
    p.add_argument("--train_file", type=str, default=None, help="Defaults to data/train_fold{fold}.jsonl")
    p.add_argument("--val_file", type=str, default=None, help="Defaults to data/val_fold{fold}.jsonl")
    p.add_argument("--output_dir", type=str, default="./medgemma_dental_lora")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # NEW ARGUMENT: Space-separated list of target modules
    p.add_argument("--lora_target_modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                   help="List of modules to target with LoRA (e.g., --lora_target_modules q_proj v_proj)")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=0, help="Fold index, only used for logging/output-dir naming.")
    p.add_argument("--early_stopping_patience", type=int, default=3, help="Stop training if eval_loss doesn't improve for N epochs.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Quantization + model loading (QLoRA, 4-bit NF4)
# --------------------------------------------------------------------------- #
def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",  # Use Flash Attention / SDPA for faster, high-end GPU training
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False  # required when gradient checkpointing is on

    return model, tokenizer


# --------------------------------------------------------------------------- #
# LoRA config
# --------------------------------------------------------------------------- #
def build_lora_model(model, r, alpha, dropout, target_modules):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,  # Updated to use the variable
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# --------------------------------------------------------------------------- #
# Dataset loading + validation of the required JSON output schema
#
# FIX: this used to list the OLD 7-field schema (Main appeal, Present medical
# history, Oral Check, Diagnosis, Treatment plan, Handle, Doctor advices).
# The current data pipeline (matching generate_predictions_new.py) only ever
# puts these 5 fields in the assistant's JSON target - Main appeal and
# Present medical history live in the USER message as query context instead,
# not in the target the model is trained to produce. Validating against the
# old 7-field list rejected every example. Updated to the 5-field schema.
# --------------------------------------------------------------------------- #
REQUIRED_FIELDS = [
    "Oral Check", "Diagnosis", "Treatment plan", "Handle", "Doctor advices",
]


def validate_example(example):
    """Sanity-check that the assistant turn is valid JSON with all 5 keys."""
    assistant_msg = example["messages"][-1]
    assert assistant_msg["role"] == "assistant", (
        f"case_id={example.get('case_id')}: last message must be the assistant turn"
    )
    try:
        parsed = json.loads(assistant_msg["content"])
    except json.JSONDecodeError as e:
        raise ValueError(
            f"case_id={example.get('case_id')}: assistant content is not valid JSON: {e}"
        )
    missing = [f for f in REQUIRED_FIELDS if f not in parsed or not str(parsed[f]).strip()]
    if missing:
        raise ValueError(
            f"case_id={example.get('case_id')}: missing/empty fields in target: {missing}"
        )
    return example


def load_and_validate(train_file, val_file):
    train_ds = load_dataset("json", data_files=train_file)["train"]
    val_ds = load_dataset("json", data_files=val_file)["train"]

    for ds, name in [(train_ds, "train"), (val_ds, "val")]:
        for ex in ds:
            validate_example(ex)
    print(f"Validated {len(train_ds)} train / {len(val_ds)} val examples "
          f"(all assistant turns are well-formed 5-field JSON).")
    return train_ds, val_ds


# --------------------------------------------------------------------------- #
# Preprocessing for modern TRL
# --------------------------------------------------------------------------- #
def convert_to_prompt_completion(example, tokenizer):
    """
    Modern TRL (>=0.12) relies on 'prompt' and 'completion' columns and uses
    `completion_only_loss=True` in SFTConfig.
    This natively handles the loss masking without a custom DataCollator.
    """
    messages = example["messages"]
    prompt_msgs = messages[:-1]

    # Render the input prompt (System + User), leaving the generation turn active
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs,
        tokenize=False,
        add_generation_prompt=True
    )

    # Render the entire conversation
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Safely slice the completion text from the full string so loss is isolated perfectly
    if full_text.startswith(prompt_text):
        completion_text = full_text[len(prompt_text):]
    else:
        # Fallback if there is a highly unusual template mismatch
        completion_text = messages[-1]["content"] + tokenizer.eos_token

    return {"prompt": prompt_text, "completion": completion_text}


# --------------------------------------------------------------------------- #
# Plotting function
# --------------------------------------------------------------------------- #
def plot_loss(log_history, output_dir, fold):
    """Extracts loss history from trainer and saves a plot."""
    train_epochs, train_losses = [], []
    eval_epochs, eval_losses = [], []

    for log in log_history:
        if "loss" in log and "epoch" in log:
            train_epochs.append(log["epoch"])
            train_losses.append(log["loss"])
        elif "eval_loss" in log and "epoch" in log:
            eval_epochs.append(log["epoch"])
            eval_losses.append(log["eval_loss"])

    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(train_epochs, train_losses, label="Training Loss", color="blue", alpha=0.6)
    if eval_losses:
        plt.plot(eval_epochs, eval_losses, label="Validation Loss", color="red", marker="o", linewidth=2)

    plt.title(f"Training and Validation Loss (Fold {fold})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plot_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved loss graph to {plot_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = os.path.join(args.output_dir, f"fold{args.fold}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading base model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # Pass the target_modules argument here
    model = build_lora_model(model, args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules)

    train_file = args.train_file or f"data/train_fold{args.fold}.jsonl"
    val_file = args.val_file or f"data/val_fold{args.fold}.jsonl"

    print(f"Loading dataset:\n  train={train_file}\n  val={val_file}")
    train_ds, val_ds = load_and_validate(train_file, val_file)

    print("Formatting dataset into prompt/completion pairs...")
    train_ds = train_ds.map(lambda x: convert_to_prompt_completion(x, tokenizer), remove_columns=["messages", "case_id"])
    val_ds = val_ds.map(lambda x: convert_to_prompt_completion(x, tokenizer), remove_columns=["messages", "case_id"])

    # FIX: SFTTrainer (completion_only_loss=True) needs room for the target
    # tokens AFTER the prompt fits within max_length. If a prompt alone
    # already exceeds max_length, that example has no valid completion span
    # and gets silently dropped - with enough long prompts (e.g. several full
    # RAG reference cases) this can drop 100% of the dataset, which then
    # fails many layers downstream with a cryptic 'num_samples=0' error from
    # RandomSampler instead of a clear message. Check up front instead.
    def token_length(example):
        n_prompt = len(tokenizer(example["prompt"], add_special_tokens=False)["input_ids"])
        n_completion = len(tokenizer(example["completion"], add_special_tokens=False)["input_ids"])
        return n_prompt, n_prompt + n_completion

    prompt_lens, total_lens = [], []
    for ex in train_ds:
        p_len, t_len = token_length(ex)
        prompt_lens.append(p_len)
        total_lens.append(t_len)

    n_prompt_too_long = sum(1 for p in prompt_lens if p >= args.max_seq_length)
    n_total_too_long = sum(1 for t in total_lens if t > args.max_seq_length)

    print(
        f"Prompt token lengths: min={min(prompt_lens)}, max={max(prompt_lens)}, "
        f"mean={sum(prompt_lens)/len(prompt_lens):.0f} "
        f"(--max_seq_length={args.max_seq_length})"
    )
    if n_prompt_too_long > 0:
        print(
            f"WARNING: {n_prompt_too_long}/{len(prompt_lens)} training examples have a PROMPT "
            f"alone >= max_seq_length ({args.max_seq_length}) - these have no room left for the "
            f"completion and will be dropped by the trainer."
        )
    if n_total_too_long > 0:
        print(
            f"NOTE: {n_total_too_long}/{len(total_lens)} training examples exceed max_seq_length "
            f"once prompt+completion are combined."
        )
    if n_prompt_too_long == len(prompt_lens):
        raise ValueError(
            f"Every training example's prompt is >= --max_seq_length ({args.max_seq_length}) tokens, "
            f"so the trainer would be left with 0 usable examples (this is what produced the "
            f"'num_samples=0' crash). Fix by either:\n"
            f"  1) Increasing --max_seq_length to comfortably exceed the max prompt length reported "
            f"above (e.g. try {max(total_lens) + 64}), or\n"
            f"  2) Reducing how much context is packed into each prompt (fewer retrieved exemplars, "
            f"e.g. --max_exemplars 2 at data-prep time, or shorter per-field truncation)."
        )

    # Build the desired config as a plain dict first, then filter down to only
    # the keys this installed version of SFTConfig actually accepts. TRL has
    # renamed/dropped SFTConfig/TrainingArguments fields across versions (e.g.
    # 'warmup_ratio' raising TypeError on some releases) - filtering avoids
    # the script breaking every time the installed trl version shifts.
    import dataclasses

    desired_sft_kwargs = dict(
        output_dir=output_dir,
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        warmup_steps=10, 
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
        packing=False,
        seed=args.seed,
        completion_only_loss=True,
        loss_type="nll",
    )

    valid_field_names = {f.name for f in dataclasses.fields(SFTConfig)}
    dropped = sorted(set(desired_sft_kwargs) - valid_field_names)
    if dropped:
        print(f"NOTE: this trl version's SFTConfig doesn't accept these fields, skipping them: {dropped}")
    filtered_sft_kwargs = {k: v for k, v in desired_sft_kwargs.items() if k in valid_field_names}

    sft_config = SFTConfig(**filtered_sft_kwargs)

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=0.0
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[early_stopping_callback],
    )

    trainer.train()

    plot_loss(trainer.state.log_history, output_dir, args.fold)

    print(f"Saving BEST LoRA adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = trainer.evaluate()
    with open(os.path.join(output_dir, "final_eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Fold {args.fold} best eval metrics: {metrics}")


if __name__ == "__main__":
    main()
