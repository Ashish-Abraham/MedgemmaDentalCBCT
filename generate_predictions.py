"""
generate_predictions.py

Runs inference with the fine-tuned MedGemma LoRA adapter over the RAG output
JSON and writes predictions.json + submission zip.

Prompt uses ONLY retrieved_exemplars — matching prepare_dataset.py, since
demographics and imaging findings are never available in the real
validation set.

INPUT  : rag_output.json  { "10": {"case_id": "10", "retrieved_exemplars": [...], ...}, ... }
OUTPUT : predictions.json { "10": {7 fields}, ... }  ->  zipped, predictions.json at root.

Usage:
  python generate_predictions.py \
    --rag_json rag_output_exp2c_clean.json \
    --base_model google/medgemma-27b-it \
    --lora_adapter ./final_model/fold_final \
    --output_json predictions.json \
    --output_zip submission.zip
"""

import os
import re
import json
import argparse
import zipfile
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


REQUIRED_FIELDS = [
    "Main appeal", "Present medical history", "Oral Check",
    "Diagnosis", "Treatment plan", "Handle", "Doctor advices",
]

# SYSTEM_PROMPT = (
#     "You are an expert dental clinical documentation assistant trained on real patient "
#     "records from a dental hospital. You will be given several similar past patient "
#     "cases retrieved for the current patient. Using them as your only source of clinical "
#     "grounding, write this patient's own 7-field clinical record.\n\n"
#     "Rules:\n"
#     "1. Base every field on patterns actually present in the retrieved cases — do not "
#     "invent tooth numbers, diagnoses, treatments, or medications that appear nowhere in "
#     "them.\n"
#     "2. Diagnosis and Oral Check must be clinically specific (e.g. exact tooth notation "
#     "and condition), matching the terminology and level of detail used in the retrieved "
#     "cases — not vague or generic restatements.\n"
#     "3. Treatment plan, Handle, and Doctor advices must be clinically consistent with the "
#     "Diagnosis and Oral Check you write — never contradict them.\n"
#     "4. Main appeal and Present medical history should read as natural patient-reported "
#     "complaints/history, in the same concise clinical register as the retrieved cases.\n"
#     "5. Every one of the 7 fields must be filled with real clinical content. Only write "
#     "'Not recorded' if every retrieved case also leaves that field empty or absent.\n"
#     "6. Output ONLY a single valid JSON object with exactly these 7 keys, in this order: "
#     "Main appeal, Present medical history, Oral Check, Diagnosis, Treatment plan, Handle, "
#     "Doctor advices. No markdown, no commentary, no text before or after the JSON."
# )
SYSTEM_PROMPT = (
    "You are an expert dental clinical documentation assistant trained on real patient "
    "records from a dental hospital. You will be given several similar past patient "
    "cases retrieved for the current patient. Using them as your only source of clinical "
    "grounding, write this patient's own 7-field clinical record.\n\n"
    "Rules:\n"
    "1. Base every field on patterns actually present in the retrieved cases — do not "
    "invent tooth numbers, diagnoses, treatments, or medications that appear nowhere in "
    "them.\n"
    "2. Diagnosis and Oral Check must be clinically specific (e.g. exact tooth notation "
    "and condition), matching the terminology and level of detail used in the retrieved "
    "cases — not vague or generic restatements.\n"
    "3. Treatment plan, Handle, and Doctor advices must be clinically consistent with the "
    "Diagnosis and Oral Check you write — never contradict them.\n"
    "4. Main appeal and Present medical history should read as natural patient-reported "
    "complaints/history, in the same concise clinical register as the retrieved cases.\n"
    "5. Every one of the 7 fields must be filled with real clinical content. Only write "
    "'Not recorded' if every retrieved case also leaves that field empty or absent.\n"
    "6. Strict Length Constraint: Keep every field extremely brief and concise, consisting "
    "of only a single short sentence or phrase (roughly 5-15 words) but do not miss necessary details and keywords."
    "7. Output ONLY a single valid JSON object with exactly these 7 keys, in this order: "
    "Main appeal, Present medical history, Oral Check, Diagnosis, Treatment plan, Handle, "
    "Doctor advices. No markdown, no commentary, no text before or after the JSON."
)

def is_placeholder(value):
    return str(value).strip().lower() in ("", "not available", "unknown", "none", "n/a", "nan", "null")


def build_user_prompt(case, max_exemplars=5):
    exemplars = case.get("retrieved_exemplars", [])[:max_exemplars]
    ex_lines = []
    for i, ex in enumerate(exemplars):
        label = chr(65 + i)
        rec = ex.get("record", {})
        field_lines = "\n".join(f"  {field}: {rec.get(field, 'Not available')}" for field in REQUIRED_FIELDS)
        ex_lines.append(f"[Case {label}] (similarity={ex.get('similarity', 0):.3f})\n{field_lines}")

    return (
        "RETRIEVED SIMILAR CASES:\n\n" + "\n\n".join(ex_lines) +
        "\n\nWrite the 7-field JSON record with keys: Main appeal, Present medical history, "
        "Oral Check, Diagnosis, Treatment plan, Handle, Doctor advices."
    )


def extract_json(raw_text):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return {}
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([\}\]])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return {}


def apply_fallbacks(record, case):
    """If the LLM leaves a field empty, fall back to the single most similar
    retrieved exemplar's value — the only reliable signal available here."""
    exemplars = case.get("retrieved_exemplars", [])
    top_record = exemplars[0]["record"] if exemplars else {}

    for field in REQUIRED_FIELDS:
        val = str(record.get(field, "")).strip()
        if not val or is_placeholder(val):
            fallback = str(top_record.get(field, "")).strip()
            record[field] = fallback if fallback and not is_placeholder(fallback) else "Not recorded"
        else:
            record[field] = val
    return record


def load_model(base_model_name, lora_adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map="auto", torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate_record(model, tokenizer, case, max_exemplars, max_new_tokens=700):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(case, max_exemplars)},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    record = extract_json(raw_text)
    print(f"--- case {case.get('case_id')} raw LLM output ---\n{raw_text}\n")   # add this line
    record = apply_fallbacks(record, case)
    return {field: record[field] for field in REQUIRED_FIELDS}


def package_zip(predictions_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_path, arcname="predictions.json")
    print(f"Packaged submission zip: {zip_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_json", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="google/medgemma-27b-it")
    parser.add_argument("--lora_adapter", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="predictions.json")
    parser.add_argument("--output_zip", type=str, default="submission.zip")
    parser.add_argument("--max_exemplars", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=700)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading RAG output from {args.rag_json} ...")
    with open(args.rag_json, "r", encoding="utf-8") as f:
        rag_data = json.load(f)

    case_ids = list(rag_data.keys())
    if args.limit:
        case_ids = case_ids[: args.limit]
    print(f"Found {len(case_ids)} cases to process.")

    print(f"Loading model: base={args.base_model}, adapter={args.lora_adapter}")
    model, tokenizer = load_model(args.base_model, args.lora_adapter)

    predictions = {}
    num_missing = 0

    for i, case_id in enumerate(case_ids, 1):
        case = rag_data[case_id]
        try:
            record = generate_record(model, tokenizer, case, args.max_exemplars, args.max_new_tokens)
        except Exception as e:
            print(f"  [WARN] case {case_id} generation failed ({e}); using fallback record.")
            record = apply_fallbacks({}, case)
            record = {field: record[field] for field in REQUIRED_FIELDS}
            num_missing += 1

        predictions[case_id] = record
        if i % 10 == 0 or i == len(case_ids):
            print(f"  Processed {i}/{len(case_ids)} cases...")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output_json} with {len(predictions)} cases ({num_missing} used pure fallback).")

    package_zip(args.output_json, args.output_zip)


if __name__ == "__main__":
    main()
