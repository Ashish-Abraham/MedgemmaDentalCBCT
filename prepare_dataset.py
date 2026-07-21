import os
import json
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split

def parse_categories(cat_str):
    """Parse string representations of lists like "['Impacted Tooth', 'Pulpitis']" into comma-separated strings."""
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
    """Constructs the user prompt based on demographic and finding fields."""
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
    
    prompt = f"PATIENT CONTEXT:\nAge: {age} ({age_group}), Sex: {sex}\n\n"
    prompt += "FINDINGS:\n"
    prompt += f"Oral Check: {oral_check}\n"
    if diagnosis_norm:
        prompt += f"Diagnosis (normalized): {diagnosis_norm}\n"
    prompt += f"Diagnosis: {diagnosis}\n\n"
    prompt += "Write the 7-field JSON record with keys: Main appeal, Present medical history, Oral Check, Diagnosis, Treatment plan, Handle, Doctor advices."
    
    return prompt

def create_assistant_response(row):
    """Constructs the assistant response as a valid JSON string containing the 7 fields."""
    record = {
        "Main appeal": str(row.get("Main appeal", "")),
        "Present medical history": str(row.get("Present medical history", "")),
        "Oral Check": str(row.get("Oral Check", "")),
        "Diagnosis": str(row.get("Diagnosis", "")),
        "Treatment plan": str(row.get("Treatment plan", "")),
        "Handle": str(row.get("Handle", "")),
        "Doctor advices": str(row.get("Doctor advices", ""))
    }
    # Ensure missing float/NaNs are converted to empty strings in the json
    for k, v in record.items():
        if v == "nan":
            record[k] = ""
            
    return json.dumps(record, ensure_ascii=False)

def main():
    csv_path = "/content/MedgemmaDentalCBCT/mmdental_cleaned_full.csv"
    output_dir = "data"
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_excel(csv_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    SYSTEM_PROMPT = (
        "You are a dental clinical documentation assistant. Given a patient's demographics and exam findings/diagnosis, "
        "write the full 7-field clinical record in this clinic's style. "
        "Copy Oral Check and Diagnosis verbatim from the findings provided; "
        "write the remaining fields to be clinically consistent with them."
    )
    
    dataset_items = []
    
    print("Formatting prompts...")
    for idx, row in df.iterrows():
        case_id = str(row.get("Filename", idx))
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": create_user_prompt(row)},
            {"role": "assistant", "content": create_assistant_response(row)}
        ]
        
        dataset_items.append({
            "case_id": case_id,
            "messages": messages
        })
        
    dataset_items = np.array(dataset_items)
    
    # ---------------------------------------------------------
    # 1. Generate Full Train/Val Split (80/20)
    # ---------------------------------------------------------
    print("\nCreating full train/val split (80/20)...")
    train_full, val_full = train_test_split(dataset_items, test_size=0.2, random_state=42)
    
    with open(os.path.join(output_dir, "train_full.jsonl"), "w", encoding="utf-8") as f:
        for item in train_full:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(os.path.join(output_dir, "val_full.jsonl"), "w", encoding="utf-8") as f:
        for item in val_full:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"Saved full split: {len(train_full)} train, {len(val_full)} val samples.")
    
    # ---------------------------------------------------------
    # 2. Generate K-Fold Splits
    # ---------------------------------------------------------
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    print(f"\nSplitting into {n_splits} folds...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset_items)):
        train_items = dataset_items[train_idx]
        val_items = dataset_items[val_idx]
        
        train_file = os.path.join(output_dir, f"train_fold{fold}.jsonl")
        val_file = os.path.join(output_dir, f"val_fold{fold}.jsonl")
        
        with open(train_file, "w", encoding="utf-8") as f:
            for item in train_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        with open(val_file, "w", encoding="utf-8") as f:
            for item in val_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                
        print(f"Fold {fold}: Saved {len(train_items)} train and {len(val_items)} val samples.")

    print("\nDataset preparation complete! Files are saved in the 'data/' directory.")

if __name__ == "__main__":
    main()
