"""
generate_predictions.py

Runs inference with the fine-tuned MedGemma LoRA adapter over the RAG output
JSON and writes predictions.json + submission zip.

Schema: only these 5 fields are generated and scored.
Main appeal / Present medical history are used as prompt CONTEXT ONLY.

Includes a copy-detection retry: if a first-pass generation is a near-verbatim
copy of a retrieved exemplar (the failure mode observed in practice), the
model is re-prompted once with sampling and an explicit corrective note
before falling back to the exemplar. A generation_status.csv sidecar records
whether each case's output was genuinely generated, generated-after-retry,
or a fallback copy, so evaluation sheets never silently mix the three.
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
    "IMPORTANT ARCHITECTURAL CONTEXT: you cannot see the patient's CBCT volume "
    "directly. A separate retrieval system has already searched for the past cases "
    "whose CBCT scans are most visually similar to the current patient's scan. The "
    "retrieved cases below are therefore your radiological evidence — they stand in "
    "for what the CBCT scan shows, in the same way a radiologist's report stands in "
    "for the image itself. Your job is to reason over this evidence, not ignore it.\n\n"
    "YOUR TASK: cross-reference the current patient's own chief complaint, age, sex, "
    "and history against the clinical findings in the retrieved cases, and deduce the "
    "most clinically consistent assessment for THIS patient. For example: if the "
    "patient's chief complaint points to pain in the upper right, and one or more "
    "retrieved cases show caries or pulpitis on an upper-right tooth, that tooth and "
    "condition are very likely the source of this patient's complaint — adopt that "
    "finding, its FDI tooth number, its ICD-10 code, and its typical treatment path as "
    "the basis for your assessment.\n\n"
    "HOW TO WEIGH MULTIPLE RETRIEVED CASES:\n"
    "- Retrieved cases are ranked by scan similarity, not by relevance to this "
    "patient's complaint. Give the most weight to whichever retrieved case(s) best "
    "match the CURRENT patient's own chief complaint, age, and history — not "
    "automatically the top-ranked one.\n"
    "- Do not adopt an entire retrieved record wholesale. If a retrieved case's tooth, "
    "condition, or procedure doesn't fit this patient's own complaint/history, discard "
    "that part of it rather than including it anyway.\n"
    "- If different retrieved cases point to different teeth/conditions, pick the one "
    "most consistent with the patient's own chief complaint; only combine findings "
    "from multiple cases if the patient's complaint plausibly involves more than one "
    "issue.\n\n"
    "FACTS vs. PHRASING — this is the critical distinction:\n"
    "- DO extract and reuse raw clinical facts from the retrieved cases: FDI tooth "
    "numbers, ICD-10 codes, anatomical conditions, and standard procedural steps, "
    "whenever they fit the current patient's complaint/history.\n"
    "- DO NOT reuse the retrieved cases' sentences. Never copy a run of 4 or more "
    "consecutive words from any retrieved case's text into your answer. Re-express the "
    "adopted facts in your own concise, telegraphic clinical phrasing, written fresh "
    "for THIS patient's record.\n"
    "- In short: copy the data, never the sentence.\n\n"
    "STYLE RULES:\n"
    "- Write in short, telegraphic clinical fragments, not full flowing sentences.\n"
    "- Every tooth reference must use the exact format 'tooth NN (anatomical position)', "
    "e.g. 'tooth 36 (lower left first molar)'. Never use '#NN' or a bare number alone.\n"
    "- When a retrieved case's condition matches the deduced diagnosis, adopt its "
    "ICD-10 code and format it as 'condition name (ICD-10)'. Do not invent a code that "
    "doesn't appear in any retrieved case.\n"
    "- Treatment plan, Handle, and Doctor advices must stay clinically consistent with "
    "the Diagnosis and Oral Check you write for THIS record, and should follow the "
    "typical procedural pattern shown in the matching retrieved case(s), rewritten in "
    "your own phrasing.\n"
    "- Every field must contain real clinical content. Only write 'Not recorded' if "
    "no retrieved case offers any relevant evidence for that field.\n\n"
    # "LENGTH: each field is one short clinical fragment, roughly 5-20 words. Do not pad, "
    # "but do not omit tooth numbers, codes, or key clinical terms.\n\n"
    "LENGTH: Keep each field concise and telegraphic. If multiple teeth or conditions are "
    "involved, you must describe all relevant findings to ensure a complete record, but do not " 
    "pad with unnecessary conversational text. "
    "OUTPUT FORMAT: output ONLY a single JSON object with exactly these 5 keys, in this "
    "order: Oral Check, Diagnosis, Treatment plan, Handle, Doctor advices. No markdown, "
    "no commentary, no text before or after the JSON."
)

RETRY_CORRECTION_NOTE = (
    "\n\nIMPORTANT: your previous draft for this patient copied a retrieved case almost "
    "word-for-word instead of describing THIS patient's own chief complaint and findings. "
    "Discard that draft. Write a new assessment grounded ONLY in the CURRENT patient's "
    "own context above, using the retrieved cases for style/terminology only."
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


def build_user_prompt(case, max_exemplars=5, correction=False):
    query_ctx = format_query_context(case)

    query_block = (
        "CURRENT PATIENT — this is who you are writing the assessment for:\n"
        f"  Age: {query_ctx['Age']}\n"
        f"  Sex: {query_ctx['Sex']}\n"
        f"  Main appeal: {query_ctx['Main appeal']}\n"
        f"  Present medical history: {query_ctx['Present medical history']}\n"
        f"  Past medical history: {query_ctx['Past medical history']}"
    )

    exemplars = case.get("retrieved_exemplars", [])[:max_exemplars]
    ex_lines = []
    for i, ex in enumerate(exemplars):
        label = chr(65 + i)
        rec = ex.get("record", {})
        field_lines = "\n".join(
            f"  {field}: {rec.get(field, 'Not available')}" for field in TARGET_FIELDS
        )
        ex_lines.append(f"[Reference Case {label}] (similarity={ex.get('similarity', 0):.3f})\n{field_lines}")
    exemplar_block = (
        "REFERENCE CASES (style/terminology guidance only — these are OTHER patients, "
        "not the one you are reporting on):\n\n" + "\n\n".join(ex_lines)
    )

    prompt = (
        f"{query_block}\n\n{exemplar_block}\n\n"
        "Write the 5-field JSON assessment for the CURRENT patient above with keys: "
        "Oral Check, Diagnosis, Treatment plan, Handle, Doctor advices."
    )
    if correction:
        prompt += RETRY_CORRECTION_NOTE
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


def get_4grams(text):
    tokens = re.sub(r"\s+", " ", str(text).strip().lower()).split()
    return set(tuple(tokens[i:i + 4]) for i in range(len(tokens) - 3))


def record_copy_ratio(record, exemplars):
    """Fraction of the generated record's 4-grams that appear verbatim in ANY
    single retrieved exemplar. High ratio = parroting one exemplar wholesale
    rather than describing the current patient. Used to trigger a retry."""
    gen_text = " ".join(str(record.get(f, "")) for f in TARGET_FIELDS)
    gen_ngrams = get_4grams(gen_text)
    if not gen_ngrams:
        return 0.0

    max_ratio = 0.0
    for ex in exemplars:
        rec = ex.get("record", {})
        ex_text = " ".join(str(rec.get(f, "")) for f in TARGET_FIELDS)
        ex_ngrams = get_4grams(ex_text)
        if not ex_ngrams:
            continue
        overlap = len(gen_ngrams & ex_ngrams) / len(gen_ngrams)
        max_ratio = max(max_ratio, overlap)
    return max_ratio


def apply_fallbacks(record, case):
    exemplars = case.get("retrieved_exemplars", [])
    top_record = exemplars[0]["record"] if exemplars else {}
    for field in TARGET_FIELDS:
        val = str(record.get(field, "")).strip()
        val = normalize_tooth_notation(val)
        if not val or is_placeholder(val):
            fallback = str(top_record.get(field, "")).strip()
            fallback = normalize_tooth_notation(fallback)
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
def _run_generation(model, tokenizer, case, max_exemplars, max_new_tokens, correction, sample):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(case, max_exemplars, correction=correction)},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
    )
    if sample:
        gen_kwargs.update(do_sample=True, temperature=0.7, top_p=0.9)
    else:
        gen_kwargs.update(do_sample=False, num_beams=4, length_penalty=0.9)

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return raw_text


def generate_record(model, tokenizer, case, max_exemplars, max_new_tokens=400, copy_threshold=0.5):
    """Returns (record_dict, status) where status is one of:
    'model', 'model_after_retry', 'fallback'."""
    exemplars = case.get("retrieved_exemplars", [])

    raw_text = _run_generation(model, tokenizer, case, max_exemplars, max_new_tokens,
                                correction=False, sample=False)
    record = extract_json(raw_text)
    parsed_ok = bool(record) and any(str(record.get(f, "")).strip() for f in TARGET_FIELDS)

    if parsed_ok and record_copy_ratio(record, exemplars) < copy_threshold:
        record = apply_fallbacks(dict(record), case)
        return record, "model"

    # Retry once: corrective note + sampling, to break out of verbatim copying.
    raw_text_retry = _run_generation(model, tokenizer, case, max_exemplars, max_new_tokens,
                                      correction=True, sample=True)
    record_retry = extract_json(raw_text_retry)
    parsed_ok_retry = bool(record_retry) and any(str(record_retry.get(f, "")).strip() for f in TARGET_FIELDS)

    if parsed_ok_retry and record_copy_ratio(record_retry, exemplars) < copy_threshold:
        record_retry = apply_fallbacks(dict(record_retry), case)
        return record_retry, "model_after_retry"

    # Both attempts failed to produce a grounded, non-copied record.
    fallback_record = apply_fallbacks({}, case)
    return fallback_record, "fallback"


def package_zip(predictions_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_path, arcname="predictions.json")
    print(f"Packaged submission zip: {zip_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_json", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="google/medgemma-27b-it")
    parser.add_argument("--lora_adapter", type=str, required=False)
    parser.add_argument("--output_json", type=str, default="predictions.json")
    parser.add_argument("--output_zip", type=str, default="submission.zip")
    parser.add_argument("--status_csv", type=str, default="generation_status.csv")
    parser.add_argument("--max_exemplars", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=400)
    parser.add_argument("--copy_threshold", type=float, default=0.5)
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
    status_rows = []
    status_counts = {"model": 0, "model_after_retry": 0, "fallback": 0}

    for i, case_id in enumerate(case_ids, 1):
        case = rag_data[case_id]
        try:
            record, status = generate_record(
                model, tokenizer, case, args.max_exemplars, args.max_new_tokens, args.copy_threshold
            )
        except Exception as e:
            print(f"  [ERROR] case {case_id} generation raised an exception:")
            traceback.print_exc()
            record = apply_fallbacks({}, case)
            status = "fallback"

        predictions[case_id] = {field: record[field] for field in TARGET_FIELDS}
        status_rows.append({"Case ID": case_id, "generation_status": status})
        status_counts[status] += 1

        if i % 10 == 0 or i == len(case_ids):
            print(f"  Processed {i}/{len(case_ids)} cases...")

    import pandas as pd
    
    # Format predictions into strings and prepare rows for Excel
    excel_rows = []
    for case_id, record in predictions.items():
        report_text = "\n".join([f"{k}: {v}" for k, v in record.items()])
        excel_rows.append({
            "Case ID": case_id,
            "LLM-Generated Report": report_text
        })
        
    output_xlsx = args.output_json.replace(".json", ".xlsx")
    pd.DataFrame(excel_rows).to_excel(output_xlsx, index=False)

    # Write status tracking CSV
    with open(args.status_csv, "w", encoding="utf-8") as f:
        f.write("Case ID,generation_status\n")
        for row in status_rows:
            f.write(f"{row['Case ID']},{row['generation_status']}\n")

    print(f"\nWrote {output_xlsx} with {len(predictions)} cases.")
    print(f"Generation status: {status_counts['model']} model, "
          f"{status_counts['model_after_retry']} model_after_retry, "
          f"{status_counts['fallback']} fallback.")
    print(f"Per-case breakdown saved to {args.status_csv} — check this before trusting metrics.")
    
    # Commented out zip packaging since output is now an Excel file
    # package_zip(args.output_json, args.output_zip)


if __name__ == "__main__":
    main()
