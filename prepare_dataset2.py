import os
import json
import ast
import argparse
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split

def parse_categories(cat_str):
    if pd.isna(cat_str):
        return ""
    cat_str = str(cat_str).strip()
    if not cat_str:
        return ""
    if cat_str.startswith("[") and cat_str.endswith("]"):
        try:
            cats = ast.literal_eval(cat_str)
            if isinstance(cats, list):
                return ", ".join([str(c) for c in cats])
        except Exception:
            pass
    return cat_str

def create_user_prompt(row):
    """Constructs the user prompt with randomized phrasing to prevent memorization."""
    age = str(row.get("Age", "Unknown"))
    age_group = str(row.get("Age_group", "Unknown"))
    if pd.isna(row.get("Age")):
        age = "Unknown"
    if pd.isna(row.get("Age_group")) or not age_group:
        age_group = "Unknown"
        
    sex = str(row.get("Sex", "Unknown"))
    if pd.isna(row.get("Sex")) or not sex:
        sex = "Unknown"
        
    oral_check = str(row.get("Oral Check", "")).strip()
    diagnosis = str(row.get("Diagnosis", "")).strip()
    diagnosis_norm = parse_categories(row.get("Diagnosis_categories", ""))
    
    # Text Augmentation: Randomize the prompt structure so the model doesn't overfit to one format
    templates = [
        # Template 1: Original strict format
        f"PATIENT CONTEXT:\nAge: {age} ({age_group}), Sex: {sex}\n\nFINDINGS:\nOral Check: {oral_check}\nDiagnosis (normalized): {diagnosis_norm}\nDiagnosis: {diagnosis}\n\nWrite the 7-field JSON record with keys: Main appeal, Present medical history, Oral Check, Diagnosis, Treatment plan, Handle, Doctor advices.",
        
        # Template 2: Narrative demographic format
        f"Patient Demographics: A {age}-year-old {sex} ({age_group}).\n\nClinical Findings:\n- Oral Check: {oral_check}\n- Diagnosis: {diagnosis} (Categories: {diagnosis_norm})\n\nPlease generate the standard 7-field clinical JSON record.",
        
        # Template 3: Direct instructional format
        f"Generate a 7-field JSON clinical record for the following case.\n\nSex: {sex}\nAge: {age}\n\nExamination Notes:\nOral Check: {oral_check}\nDiagnosis: {diagnosis}"
    ]
    
    return random.choice(templates)

def create_assistant_response(row):
    record = {
        "Main appeal": str(row.get("Main appeal", "")).strip(),
        "Present medical history": str(row.get("Present medical history", "")).strip(),
        "Oral Check": str(row.get("Oral Check", "")).strip(),
        "Diagnosis": str(row.get("Diagnosis", "")).strip(),
        "Treatment plan": str(row.get("Treatment plan", "")).strip(),
        "Handle": str(row.get("Handle", "")).strip(),
        "Doctor advices": str(row.get("Doctor advices", "")).strip()
    }
    
    for k, v in record.items():
        if v.lower() == "nan" or not v:
            record[k] = "Not recorded"
            
    return json.dumps(record, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for MedGemma LoRA Training")
    parser.add_argument("--csv_path", type=str, default="/root/MedgemmaDentalCBCT/mmdental_cleaned_full.csv")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--split", type=float, default=0.1)
    args = parser.parse_args()
    
    print(f"Loading data from {args.csv_path}...")
    df = pd.read_excel(args.csv_path)
    os.makedirs(args.output_dir, exist_ok=True)
    
    SYSTEM_PROMPT = (
        "You are a dental clinical documentation assistant. Given the patient's demographics and exam findings/diagnosis, "
        "generate a complete 7-field clinical record in the clinic's style. Copy 'Oral Check' and 'Diagnosis' exactly as "
        "provided, without paraphrasing or omitting clinical details (including tooth notation and terminology). Write the "
        "remaining fields so they are clinically specific, internally consistent, and aligned with the findings. 'Main appeal' "
        "and 'Present medical history' should be concise, natural patient-reported statements. Populate every field with "
        "meaningful clinical content unless the information is genuinely unavailable. Output ONLY a single valid JSON object "
        "with exactly these keys, in this order: Main appeal, Present medical history, Oral Check, Diagnosis, Treatment plan, "
        "Handle, Doctor advices."
    )
    
    dataset_items = []
    print("Formatting prompts with randomized structures...")
    for idx, row in df.iterrows():
        case_id = str(row.get("Filename", idx))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": create_user_prompt(row)},
            {"role": "assistant", "content": create_assistant_response(row)}
        ]
        dataset_items.append({"case_id": case_id, "messages": messages})
        
    dataset_items = np.array(dataset_items)
    
    # 1. Generate Full Train/Val Split (90/10)
    train_full, val_full = train_test_split(dataset_items, test_size=args.split, random_state=42)
    
    with open(os.path.join(args.output_dir, "train_full.jsonl"), "w", encoding="utf-8") as f:
        for item in train_full:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(os.path.join(args.output_dir, "val_full.jsonl"), "w", encoding="utf-8") as f:
        for item in val_full:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Saved full split: {len(train_full)} train, {len(val_full)} val samples.")
    
    # 2. Generate K-Fold Splits (10 Folds)
    n_splits = 10
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"\nSplitting into {n_splits} folds...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_items)):
        train_items = dataset_items[train_idx]
        val_items = dataset_items[val_idx]
        
        train_file = os.path.join(args.output_dir, f"train_fold{fold}.jsonl")
        val_file = os.path.join(args.output_dir, f"val_fold{fold}.jsonl")
        
        with open(train_file, "w", encoding="utf-8") as f:
            for item in train_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        with open(val_file, "w", encoding="utf-8") as f:
            for item in val_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        print(f"Fold {fold}: Saved {len(train_items)} train and {len(val_items)} val samples.")

if __name__ == "__main__":
    main()
