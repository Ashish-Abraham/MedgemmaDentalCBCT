"""
generate_predictions.py

Runs inference with the fine-tuned MedGemma LoRA adapter over the RAG output
JSON (one entry per validation case, produced by the Stage-1 retrieval
pipeline) and writes a submission-ready predictions.json + zip.

INPUT  : rag_output.json
  {
    "88": {
      "case_id": "88",
      "patient_context": {...},
      "predicted_findings": {
        "tooth_notations": [...], "tooth_notations_expanded": [...],
        "diagnosis_codes": [...], "treatment_actions": [...],
        "medications": [...], "confidence": {...}
      },
      "draft_text": {"oral_check_draft": "...", "diagnosis_draft": "...", "treatment_plan_draft": "..."},
      "retrieval_scores": {...},
      "retrieved_exemplars": [ {"case_id": "...", "similarity": 0.82, "record": {7 fields}}, ... ]
    },
    "378": {...},
    ...
  }

OUTPUT : predictions.json
  {
    "88":  {"Main appeal": "...", "Present medical history": "...", "Oral Check": "...",
            "Diagnosis": "...", "Treatment plan": "...", "Handle": "...", "Doctor advices": "..."},
    "378": {...}
  }
  packaged into <output_zip> with predictions.json at the root (no subfolder).

Usage:
  python generate_predictions.py \
    --rag_json rag_output_exp2c_clean.json \
    --base_model google/medgemma-27b-it \
    --lora_adapter ./final_model/fold_final \
    --output_json predictions.json \
    --output_zip submission.zip \
    --max_exemplars 3
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

SYSTEM_PROMPT = (
    "You are a dental clinical documentation assistant. Given a patient's demographics and "
    "exam findings/diagnosis, write the full 7-field clinical record in this clinic's style. "
    "Copy Oral Check and Diagnosis verbatim from the findings provided; write the remaining "
    "fields to be clinically consistent with them."
)
# ^ Identical to the system prompt used in prepare_dataset.py's training data.
# Do not change this without retraining — the model was fine-tuned to this exact instruction.


def dedup_phrase_list(text):
    """
    predicted draft text is often a noisy concatenation of repeated per-tooth
    predictions (e.g. 'malformed teeth malformed teeth malformed teeth').
    Collapse consecutive/duplicate phrases so the FINDINGS block reads closer
    to natural clinical text, matching what the model saw during training.
    """
    if not text:
        return text
    parts = [p.strip() for p in re.split(r"[,.;]\s*", text) if p.strip()]
    seen = set()
    deduped = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return ". ".join(deduped)


# --------------------------------------------------------------------------- #
# Prompt construction — matches Template 1 from prepare_dataset.py exactly,
# with retrieved exemplars appended as a NEW trailing section (additive only,
# never restructuring the part the model was actually trained on).
# --------------------------------------------------------------------------- #
def format_exemplar(idx_label, exemplar):
    rec = exemplar.get("record", {})
    lines = [f"[{idx_label}] (similarity={exemplar.get('similarity', 0):.3f})"]
    for field in REQUIRED_FIELDS:
        val = str(rec.get(field, "")).strip() or "Not available"
        lines.append(f"  {field}: {val}")
    return "\n".join(lines)


def build_user_prompt(case, max_exemplars=3, include_exemplars=True):
    ctx = case.get("patient_context", {})
    age = ctx.get("age", "Unknown")
    age_group = ctx.get("age_group", "Unknown")
    sex = ctx.get("sex", "Unknown")

    findings = case.get("predicted_findings", {})
    diagnosis_norm = ", ".join(str(d) for d in findings.get("diagnosis_codes", []))
    draft = case.get("draft_text", {})

    oral_check = dedup_phrase_list(draft.get("oral_check_draft", "").strip()) or "Not available"
    diagnosis = dedup_phrase_list(draft.get("diagnosis_draft", "").strip()) or "Not available"

    # --- exact Template 1 shape from prepare_dataset.py ---
    prompt = (
        f"PATIENT CONTEXT:\nAge: {age} ({age_group}), Sex: {sex}\n\n"
        f"FINDINGS:\nOral Check: {oral_check}\n"
        f"Diagnosis (normalized): {diagnosis_norm}\nDiagnosis: {diagnosis}\n\n"
        f"Write the 7-field JSON record with keys: Main appeal, Present medical history, "
        f"Oral Check, Diagnosis, Treatment plan, Handle, Doctor advices."
    )

    if include_exemplars:
        exemplars = case.get("retrieved_exemplars", [])[:max_exemplars]
        if exemplars:
            exemplar_block = "\n\n".join(
                format_exemplar(f"Case {chr(65+i)}", ex) for i, ex in enumerate(exemplars)
            )
            # Appended AFTER the trained instruction, framed so it can only add
            # information (phrasing style) without contradicting the FINDINGS above.
            prompt += (
                "\n\nFor additional style/phrasing reference only, here are similar past "
                "cases. Do NOT copy any tooth number, diagnosis, or treatment detail from "
                "them that conflicts with the FINDINGS above — only borrow tone and phrasing "
                "for fields not covered by the findings (Main appeal, Present medical "
                "history, Handle, Doctor advices):\n\n" + exemplar_block
            )

    return prompt


# --------------------------------------------------------------------------- #
# JSON extraction / repair from raw LLM output
# --------------------------------------------------------------------------- #
def extract_json(raw_text):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return {}
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # common repair: strip trailing commas before } or ]
        repaired = re.sub(r",\s*([\}\]])", r"\1", candidate)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return {}


# --------------------------------------------------------------------------- #
# Fallback filling for missing/empty fields (protects field_completion_rate)
# --------------------------------------------------------------------------- #
def apply_fallbacks(record, case):
    draft = case.get("draft_text", {})
    exemplars = case.get("retrieved_exemplars", [])
    top_exemplar_record = exemplars[0]["record"] if exemplars else {}

    fallback_map = {
        "Oral Check": draft.get("oral_check_draft", ""),
        "Diagnosis": draft.get("diagnosis_draft", ""),
        "Treatment plan": draft.get("treatment_plan_draft", ""),
    }

    for field in REQUIRED_FIELDS:
        val = str(record.get(field, "")).strip()
        if not val or val.lower() in ("nan", "not recorded", "none"):
            fallback = fallback_map.get(field, "") or str(top_exemplar_record.get(field, "")).strip()
            record[field] = fallback if fallback else "Not recorded"
        else:
            record[field] = val
    return record


# --------------------------------------------------------------------------- #
# Entity consistency enforcement — SOFT by default.
#
# IMPORTANT TRADE-OFF: draft_text is a noisy, often repetitive concatenation
# of per-tooth predictions (e.g. "malformed teeth malformed teeth malformed
# teeth"). Overwriting fluent, fine-tuned LLM output with this raw text
# whenever it merely paraphrases will actively HURT BLEU/METEOR/ROUGE-L/CIDEr
# for a gain on entity F1 that isn't guaranteed (the noisy text isn't
# necessarily easier for the grader's entity extractor either).
#
# Default behavior: only intervene when the LLM field is genuinely
# empty/degenerate. Log a warning on suspected mismatches instead of
# silently overwriting, so you can inspect how often this actually happens
# on your local eval before deciding whether strict mode is worth it.
# --------------------------------------------------------------------------- #
DEGENERATE_VALUES = {"", "nan", "not available", "not recorded", "none", "n/a"}


def is_degenerate(value):
    return str(value).strip().lower() in DEGENERATE_VALUES


def enforce_entity_consistency(record, case, strict=False, verbose=True):
    draft = case.get("draft_text", {})
    findings = case.get("predicted_findings", {})
    case_id = case.get("case_id", "?")

    diagnosis_codes = [str(d).strip() for d in findings.get("diagnosis_codes", []) if str(d).strip()]
    llm_diag = record.get("Diagnosis", "")
    if diagnosis_codes and not is_degenerate(llm_diag):
        matched = any(code.lower() in llm_diag.lower() for code in diagnosis_codes)
        if not matched:
            if verbose:
                print(f"  [NOTE] case {case_id}: Diagnosis may not match predicted codes "
                      f"{diagnosis_codes} — LLM output kept as-is (soft mode).")
            if strict:
                record["Diagnosis"] = dedup_phrase_list(draft.get("diagnosis_draft", llm_diag))
    elif is_degenerate(llm_diag):
        # genuinely empty — safe to fill from the grounded draft regardless of mode
        record["Diagnosis"] = dedup_phrase_list(draft.get("diagnosis_draft", "")) or llm_diag

    tooth_expanded = findings.get("tooth_notations_expanded", [])
    llm_oral = record.get("Oral Check", "")
    if tooth_expanded and not is_degenerate(llm_oral):
        matched = any(t.split(" (")[0].lower() in llm_oral.lower() for t in tooth_expanded)
        if not matched:
            if verbose:
                print(f"  [NOTE] case {case_id}: Oral Check may not mention predicted teeth "
                      f"{[t.split(' (')[0] for t in tooth_expanded]} — LLM output kept as-is (soft mode).")
            if strict:
                record["Oral Check"] = dedup_phrase_list(draft.get("oral_check_draft", llm_oral))
    elif is_degenerate(llm_oral):
        record["Oral Check"] = dedup_phrase_list(draft.get("oral_check_draft", "")) or llm_oral

    return record


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(base_model_name, lora_adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    model.eval()
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Single-case generation
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def generate_record(model, tokenizer, case, max_exemplars, max_new_tokens=700,
                     include_exemplars=True, strict_entities=False):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(case, max_exemplars, include_exemplars)},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        num_beams=1,
    )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    record = extract_json(raw_text)
    record = apply_fallbacks(record, case)
    record = enforce_entity_consistency(record, case, strict=strict_entities)
    # final key-order normalization to match the submission spec exactly
    return {field: record[field] for field in REQUIRED_FIELDS}


# --------------------------------------------------------------------------- #
# Zip packaging (predictions.json at zip root, no subfolder)
# --------------------------------------------------------------------------- #
def package_zip(predictions_path, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_path, arcname="predictions.json")
    print(f"Packaged submission zip: {zip_path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag_json", type=str, required=True,
                         help="Path to the RAG output JSON (rag_output_exp2c_clean.json)")
    parser.add_argument("--base_model", type=str, default="google/medgemma-27b-it")
    parser.add_argument("--lora_adapter", type=str, required=True,
                         help="Path to the final fine-tuned LoRA adapter directory")
    parser.add_argument("--output_json", type=str, default="predictions.json")
    parser.add_argument("--output_zip", type=str, default="submission.zip")
    parser.add_argument("--max_exemplars", type=int, default=3,
                         help="Max retrieved exemplars to include in the prompt")
    parser.add_argument("--max_new_tokens", type=int, default=700)
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional: only process first N cases (debugging)")
    parser.add_argument("--no_exemplars", action="store_true",
                         help="Disable the retrieved-exemplars addendum (pure fine-tuned-format prompt, "
                              "closest to what the model was actually trained on — good baseline to "
                              "compare against with-exemplars runs).")
    parser.add_argument("--strict_entities", action="store_true",
                         help="Force Diagnosis/Oral Check to the grounded (but noisier) draft text "
                              "whenever the LLM's phrasing doesn't literally contain the predicted "
                              "entities. Off by default — test both on your local k-fold eval before "
                              "choosing, since this trades fluency metrics for entity-F1 safety.")
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
            record = generate_record(
                model, tokenizer, case,
                max_exemplars=args.max_exemplars,
                max_new_tokens=args.max_new_tokens,
                include_exemplars=not args.no_exemplars,
                strict_entities=args.strict_entities,
            )
        except Exception as e:
            print(f"  [WARN] case {case_id} generation failed ({e}); using pure fallback record.")
            record = apply_fallbacks({}, case)
            record = enforce_entity_consistency(record, case, strict=args.strict_entities)
            record = {field: record[field] for field in REQUIRED_FIELDS}
            num_missing += 1

        predictions[case_id] = record
        if i % 10 == 0 or i == len(case_ids):
            print(f"  Processed {i}/{len(case_ids)} cases...")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output_json} with {len(predictions)} cases "
          f"({num_missing} used pure fallback due to generation errors).")

    package_zip(args.output_json, args.output_zip)


if __name__ == "__main__":
    main()
