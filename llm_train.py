"""
train_medgemma_lora.py

QLoRA fine-tuning of google/medgemma-27b-it for the MMDental multimodal
clinical-record generation task (MICCAI STSR 2026, Task 3).

ASSUMPTIONS (produced by prepare_dataset.py, not this file):
  - train.jsonl and val.jsonl exist, one JSON object per line, each with a
    "messages" key in HF chat format:

      {
        "case_id": "1",
        "messages": [
          {"role": "system", "content": "..."},
          {"role": "user", "content": "..."},          # Clean prompt (demographics + findings, NO RAG)
          {"role": "assistant", "content": "{...7-field JSON...}"}
        ]
      }

  - The assistant turn is a single JSON string with exactly these keys:
      Main appeal, Present medical history, Oral Check, Diagnosis,
      Treatment plan, Handle, Doctor advices

Hardware target: High-VRAM GPU (e.g. A100 80GB) for training to maximize score.
(The 24GB RTX 3090 constraint is for Docker deployment only).
Training uses pure bfloat16 (no 4-bit quantization), high-rank LoRA, and larger batch sizes.
"""

import os
import json
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM


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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=0,
                    help="Fold index, only used for logging/output-dir naming under k-fold CV.")
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
def build_lora_model(model, r, alpha, dropout):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# --------------------------------------------------------------------------- #
# Dataset loading + validation of the required JSON output schema
# --------------------------------------------------------------------------- #
REQUIRED_FIELDS = [
    "Main appeal", "Present medical history", "Oral Check",
    "Diagnosis", "Treatment plan", "Handle", "Doctor advices",
]


def validate_example(example):
    """Sanity-check that the assistant turn is valid JSON with all 7 keys.
    Raises with the case_id so a bad prepare_dataset.py output fails loudly
    instead of silently training on malformed targets."""
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
          f"(all assistant turns are well-formed 7-field JSON).")
    return train_ds, val_ds


# --------------------------------------------------------------------------- #
# Formatting function: messages -> single training string via chat template
# --------------------------------------------------------------------------- #
def make_formatting_func(tokenizer):
    def formatting_func(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,  # full conversation, incl. assistant turn, for training
        )
        return text
    return formatting_func


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
    model = build_lora_model(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    train_file = args.train_file or f"data/train_fold{args.fold}.jsonl"
    val_file = args.val_file or f"data/val_fold{args.fold}.jsonl"

    print(f"Loading dataset:\n  train={train_file}\n  val={val_file}")
    train_ds, val_ds = load_and_validate(train_file, val_file)

    formatting_func = make_formatting_func(tokenizer)

    # Only compute loss on the assistant's response tokens, not the (long)
    # system/user prompt containing demographics and clinical findings.
    # response_template must match how the chat template renders the start
    # of the assistant turn for this tokenizer/model family — verify against
    # a decoded sample before a long training run (see the printed check below).
    response_template = "<start_of_turn>model\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Quick sanity print: confirm the response_template actually appears in
    # a rendered example, otherwise the collator will silently mask nothing
    # (== training on the full prompt, which you don't want).
    sample_text = formatting_func(train_ds[0])
    if response_template not in sample_text:
        raise ValueError(
            "response_template not found in a rendered training example — "
            "check MedGemma's chat template turn markers and update "
            "response_template accordingly before training.\n\n"
            f"Rendered sample:\n{sample_text[:2000]}"
        )

    sft_config = SFTConfig(
        output_dir=output_dir,
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        optim="adamw_torch",        # Standard fast optimizer for high-end GPUs
        report_to="none",
        packing=False,               # keep each case as its own example, no cross-case packing
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        formatting_func=formatting_func,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    print(f"Saving LoRA adapter to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Final eval loss for this fold, useful when averaging across k-fold CV runs
    metrics = trainer.evaluate()
    with open(os.path.join(output_dir, "final_eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Fold {args.fold} final eval metrics: {metrics}")


if __name__ == "__main__":
    main()