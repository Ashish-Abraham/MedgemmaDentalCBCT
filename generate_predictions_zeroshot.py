"""
generate_predictions_zeroshot.py

Zero-shot version of generate_predictions_new.py: runs inference with the
fine-tuned MedGemma LoRA adapter over the RAG output JSON and writes
predictions.json, using ONLY the current patient's own context (age, sex,
chief complaint, history, draft findings). No retrieved exemplars are shown
to the model, and there is no copy-detection / retry logic, since with no
exemplars in the prompt there is nothing for the model to copy from.

Schema: only these 5 fields are generated and scored.
"""

import os
import re
import json
import argparse
import zipfile
import traceback
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

TARGET_FIELDS = ["Oral Check", "Diagnosis", "Treatment plan", "Handle", "Doctor advices"]

SYSTEM_PROMPT = (
    "You are an expert dental clinician completing the post-examination assessment "
    "section of a structured dental record for the CURRENT patient.\n\n"
    "YOUR TASK: based on the patient's chief complaint, age, sex, history, and any "
    "draft findings provided, produce the most clinically consistent assessment for "
    "THIS patient, using standard dental/radiological clinical knowledge.\n\n"
    "STYLE RULES:\n"
    "- Write in short, telegraphic clinical fragments, not full flowing sentences.\n"
    "- Every tooth reference must use the exact format 'tooth NN (anatomical position)', "
    "e.g. 'tooth 36 (lower left first molar)'. Never use '#NN' or a bare number alone.\n"
    "- When you state a diagnosis, format it as 'condition name (ICD-10)' using the "
    "standard ICD-10 code for that condition.\n"
    "- Treatment plan, Handle, and Doctor advices must stay clinically consistent with "
    "the Diagnosis and Oral Check you write for THIS record.\n"
    "- Every field must contain real clinical content. Only write 'Not recorded' if "
    "there is truly no basis to infer that field from the patient context provided.\n\n"
    "LENGTH: Keep each field concise and telegraphic. If multiple teeth or conditions are "
    "involved, you must describe all relevant findings to ensure a complete record, but do not "
    "pad with unnecessary conversational text. "
    "OUTPUT FORMAT: output ONLY a single JSON object with exactly these 5 keys, in this "
    "order: Oral Check, Diagnosis, Treatment plan, Handle, Doctor advices. No markdown, "
    "no commentary, no text before or after the JSON."
)


def is_placeholder(value):
    return str(value).strip().lower() in ("", "not available", "unknown", "none", "n/a", "nan", "null")


def format_query_context(case):
    pc = case.get("patient_context", {}) or {}
    age = pc.get("age", "unknown")
    age_group = pc.get("age_group", "")
    sex = pc.get("sex", "unknown")
    past_history = pc.get("past_medical_history", "Not recorded")
    main_appeal = (
        case.get("main_appeal") or case.get("Main appeal")
        or pc.get("main_appeal") or pc.get("Main appeal")
        or case.get("chief_complaint") or pc.get("chief_complaint")
        or "Not recorded"
    )
    present_history = (
        case.get("present_medical_history") or case.get("Present medical history")
        or pc.get("present_medical_history") or pc.get("Present medical history")
        or "Not recorded"
    )
    age_str = f"{age} ({age_group})" if age_group else str(age)
    return {
        "Age": age_str,
        "Sex": sex,
        "Main appeal": str(main_appeal).strip(),
        "Present medical history": str(present_history).strip(),
        "Past medical history": str(past_history).strip(),
    }


def build_user_prompt(case):
    query_ctx = format_query_context(case)

    drafts = case.get("draft_text", {}) or {}
    findings = case.get("predicted_findings", {}) or {}

    oral_check_draft = str(drafts.get("oral_check_draft", "")).strip() or "None"
    diagnosis_draft = str(drafts.get("diagnosis_draft", "")).strip() or "None"
    treatment_plan_draft = (
        str(drafts.get("treatment_plan_draft", "")).strip() or "None"
    )

    suspected_teeth = (
        ", ".join(findings.get("tooth_notations_expanded", []))
        or ", ".join(findings.get("tooth_notations", []))
        or "None"
    )
    detected_codes = ", ".join(findings.get("diagnosis_codes", [])) or "None"

    query_block = (
        "CURRENT PATIENT — this is who you are writing the assessment for:\n"
        f"  Age: {query_ctx['Age']}\n"
        f"  Sex: {query_ctx['Sex']}\n"
        f"  Main appeal: {query_ctx['Main appeal']}\n"
        f"  Present medical history: {query_ctx['Present medical history']}\n"
        f"  Past medical history: {query_ctx['Past medical history']}\n"
        f"  Suspected Teeth / Sites: {suspected_teeth}\n"
        f"  Draft Findings (CBCT/Exam): {oral_check_draft}\n"
        f"  Draft Diagnosis Findings: {diagnosis_draft}\n"
        f"  Draft Planned Actions: {treatment_plan_draft}\n"
        f"  Detected Finding Codes: {detected_codes}"
    )

    prompt = (
        f"{query_block}\n\n"
        "Write the 5-field JSON assessment for the CURRENT patient above with keys: "
        "Oral Check, Diagnosis, Treatment plan, Handle, Doctor advices."
    )
    return prompt


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


def normalize_tooth_notation(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"tooth\s*#\s*(\d{1,2})", r"tooth \1", text, flags=re.IGNORECASE)
    text = re.sub(r"#(\d{1,2})\b", r"tooth \1", text)
    return text


def apply_placeholder_cleanup(record):
    """No exemplar to fall back to in zero-shot mode, so we just normalize
    tooth notation and mark genuinely empty fields as 'Not recorded'."""
    for field in TARGET_FIELDS:
        val = str(record.get(field, "")).strip()
        val = normalize_tooth_notation(val)
        record[field] = val if val and not is_placeholder(val) else "Not recorded"
    return record


def load_model(base_model_name, lora_adapter_path=None):
    load_path = lora_adapter_path if lora_adapter_path else base_model_name
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    if lora_adapter_path:
        model = PeftModel.from_pretrained(model, lora_adapter_path)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def _run_generation(model, tokenizer, case, max_new_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(case)},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
        do_sample=False,
        num_beams=4,
        length_penalty=0.9,
    )

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw_text


def generate_record(model, tokenizer, case, max_new_tokens=400):
    """Returns (record_dict, status) where status is one of:
    'model', 'parse_failed'."""
    raw_text = _run_generation(model, tokenizer, case, max_new_tokens)
    record = extract_json(raw_text)
    parsed_ok = bool(record) and any(str(record.get(f, "")).strip() for f in TARGET_FIELDS)

    if parsed_ok:
        record = apply_placeholder_cleanup(dict(record))
        return record, "model"

    # Model failed to produce parseable JSON with any content; there is no
    # exemplar to fall back to in zero-shot mode, so return an empty/"Not
    # recorded" record and flag it.
    empty_record = apply_placeholder_cleanup({})
    return empty_record, "parse_failed"


def package_zip(predictions_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_path, arcname="predictions.json")
    print(f"Packaged submission zip: {zip_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_json", type=str, required=True,
                         help="Input JSON (still keyed by case_id -> case dict; "
                              "retrieved_exemplars, if present, are ignored).")
    parser.add_argument("--base_model", type=str, default="google/medgemma-1.5-4b-it")
    parser.add_argument("--lora_adapter", type=str, required=False, default=None)
    parser.add_argument("--output_json", type=str, default="predictions_zeroshot.json")
    parser.add_argument("--output_zip", type=str, default="submission_zeroshot.zip")
    parser.add_argument("--status_csv", type=str, default="generation_status_zeroshot.csv")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading input data from {args.rag_json} ...")
    with open(args.rag_json, "r", encoding="utf-8") as f:
        rag_data = json.load(f)

    case_ids = list(rag_data.keys())
    if args.limit:
        case_ids = case_ids[: args.limit]
    print(f"Found {len(case_ids)} cases to process.")

    print(f"Loading model: base={args.base_model}, adapter={args.lora_adapter}")
    model, tokenizer = load_model(args.base_model, args.lora_adapter)

    predictions = {}
    status_rows = []
    status_counts = {"model": 0, "parse_failed": 0}

    for i, case_id in enumerate(case_ids, 1):
        case = rag_data[case_id]
        try:
            record, status = generate_record(model, tokenizer, case, args.max_new_tokens)
        except Exception as e:
            print(f"  [ERROR] case {case_id} generation raised an exception:")
            traceback.print_exc()
            record = apply_placeholder_cleanup({})
            status = "parse_failed"

        predictions[case_id] = {field: record[field] for field in TARGET_FIELDS}
        status_rows.append({"Case ID": case_id, "generation_status": status})
        status_counts[status] += 1

        if i % 10 == 0 or i == len(case_ids):
            print(f"  Processed {i}/{len(case_ids)} cases...")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    with open(args.status_csv, "w", encoding="utf-8") as f:
        f.write("Case ID,generation_status\n")
        for row in status_rows:
            f.write(f"{row['Case ID']},{row['generation_status']}\n")

    print(f"\nWrote {args.output_json} with {len(predictions)} cases.")
    print(f"Generation status: {status_counts['model']} model, "
          f"{status_counts['parse_failed']} parse_failed.")

    #package_zip(args.output_json, args.output_zip)


if __name__ == "__main__":
    main()
