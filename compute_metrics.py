# """
# compute_metrics.py
# ===================
# Computes all evaluation metrics for the retrieval-grounded dental CBCT
# report-generation paper from an Excel sheet of (Ground Truth, LLM-Generated)
# report pairs, optionally with a "Retrieved Exemplars" column (JSON list, as
# returned by the retrieval module) for the retrieval-specific checks.

# METRICS COMPUTED
# -----------------
# 1. BLEU-4                          (corpus-level, sacrebleu)
# 2. ROUGE-L                         (F-measure, rouge-score)
# 3. METEOR                          (nltk)
# 4. CIDEr                           (pycocoevalcap)
# 5. FDI Tooth-Set F1                (macro over patients)
# 6. ICD-10 Set F1                   (macro over patients)
# 7. Hallucination / Factual-Consistency Rate
#    -> fraction of (tooth + ICD-10) entities asserted in the generated report
#       that are ABSENT from the ground-truth report for that patient.
#       (Reported per-patient mean and micro-pooled.)
# 8. Retrieval Copy Rate
#    -> fraction of generated 4-grams that appear in a retrieved exemplar's
#       text but do NOT appear in the ground-truth reference. High = parroting
#       retrieved exemplars rather than generating from the actual case.
#       Requires the "Retrieved Exemplars" column.
# 9. Fabricated Tooth-Reference Rate
#    -> fraction of FDI tooth numbers asserted in the generated report that
#       appear in NEITHER the ground-truth report NOR any retrieved exemplar
#       record (i.e. no evidence anywhere supports the claim). Requires the
#       "Retrieved Exemplars" column; falls back to GT-only evidence if absent.

# INPUT
# -----
# An .xlsx file with (at minimum) these columns:
#     - "Case ID"
#     - "Ground Truth Report"      (free text, same "Field: value" format used
#                                    in the paper's JSON report schema)
#     - "LLM-Generated Report"     (free text, same format)
# Optional column (needed for metrics 8 & 9):
#     - "Retrieved Exemplars"      (a JSON string: a list of objects, each with
#                                    "case_id", "similarity", ... and a "record"
#                                    dict of the same 7 report fields, exactly
#                                    as returned by the retrieval module.)

# USAGE
# -----
#     python compute_metrics.py /path/to/ground_truth_vs_llm_report.xlsx

# Outputs:
#     - metrics_per_case.csv   (per-patient breakdown for every metric)
#     - metrics_summary.csv    (aggregate table, ready to paste into the paper)
#     - Console printout of the summary table
# """

# import argparse
# import json
# import re
# import sys
# from collections import Counter

# import numpy as np
# import pandas as pd

# # ----------------------------------------------------------------------
# # Metric library imports
# # ----------------------------------------------------------------------
# import sacrebleu
# from rouge_score import rouge_scorer
# import nltk
# from nltk.translate.meteor_score import meteor_score
# from pycocoevalcap.cider.cider import Cider

# for pkg in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
#     try:
#         nltk.data.find(f"corpora/{pkg}")
#     except LookupError:
#         try:
#             nltk.data.find(f"tokenizers/{pkg}")
#         except LookupError:
#             nltk.download(pkg, quiet=True)


# # ----------------------------------------------------------------------
# # Text normalization / tokenization
# # ----------------------------------------------------------------------
# def normalize(text: str) -> str:
#     """Lowercase, collapse whitespace. Applied consistently to GT, LLM,
#     and retrieved-exemplar text so every metric sees the same tokenization."""
#     if pd.isna(text):
#         return ""
#     text = str(text)
#     text = text.replace("&#39;", "'").replace("&amp;", "&")
#     text = re.sub(r"\s+", " ", text).strip().lower()
#     return text


# def tokenize(text: str):
#     return normalize(text).split()


# def get_ngrams(tokens, n=4):
#     return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


# def all_ngrams_list(tokens, n=4):
#     """Return the ngrams as a list (not set) so repeated copied ngrams are
#     each counted once per occurrence, for the copy-rate denominator."""
#     return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# # ----------------------------------------------------------------------
# # Clinical entity extraction
# # ----------------------------------------------------------------------
# # Valid FDI tooth numbers: quadrant digit in {1..8}, tooth-position digit
# # in {1..8}. This excludes incidental two-digit numbers in the text (ages,
# # durations, percentages, etc.) that don't form a valid FDI pattern, e.g.
# # "40 minutes" (position digit 0 is invalid) is correctly excluded.
# FDI_RE = re.compile(r"\b([1-8])([1-8])\b")

# # ICD-10-style codes as used in the MMDental records, e.g. "K02.400",
# # "K08.302", "Z01.200".
# ICD10_RE = re.compile(r"\b([A-Za-z]\d{2}\.\d{1,4})\b")


# def extract_fdi_teeth(text: str) -> set:
#     text = normalize(text)
#     return {f"{a}{b}" for a, b in FDI_RE.findall(text)}


# def extract_icd10(text: str) -> set:
#     text = normalize(text)
#     return {code.upper() for code in ICD10_RE.findall(text)}


# # ----------------------------------------------------------------------
# # Set-based F1 (macro over patients)
# # ----------------------------------------------------------------------
# def set_f1(ref_set: set, hyp_set: set):
#     """Precision/recall/F1 between two entity sets for a single patient.
#     Convention for degenerate cases:
#       - both empty            -> P=R=F1=1.0 (nothing to find, nothing wrongly claimed)
#       - ref empty, hyp non-empty -> P=0, R=1.0 (undefined -> treated as N/A), F1=0
#       - ref non-empty, hyp empty -> P=1.0 (undefined -> N/A), R=0, F1=0
#     """
#     if not ref_set and not hyp_set:
#         return 1.0, 1.0, 1.0
#     if not hyp_set:
#         return np.nan, 0.0, 0.0
#     if not ref_set:
#         return 0.0, np.nan, 0.0
#     tp = len(ref_set & hyp_set)
#     precision = tp / len(hyp_set)
#     recall = tp / len(ref_set)
#     f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
#     return precision, recall, f1


# # ----------------------------------------------------------------------
# # Retrieved-exemplar parsing
# # ----------------------------------------------------------------------
# REPORT_FIELDS = [
#     "Main appeal", "Present medical history", "Oral Check", "Diagnosis",
#     "Treatment plan", "Handle", "Doctor advices",
# ]


# def parse_exemplars(cell) -> list:
#     """Parse the 'Retrieved Exemplars' cell (a JSON string) into a list of
#     dicts. Returns [] if the cell is empty/NaN or cannot be parsed."""
#     if pd.isna(cell):
#         return []
#     if isinstance(cell, list):
#         return cell
#     cell = str(cell).strip()
#     if not cell:
#         return []
#     try:
#         return json.loads(cell)
#     except json.JSONDecodeError:
#         pass
#     # fall back for python-repr-style strings (single quotes, etc.)
#     try:
#         import ast
#         return ast.literal_eval(cell)
#     except Exception:
#         print(f"  [warn] could not parse Retrieved Exemplars cell, skipping: {cell[:80]}...",
#               file=sys.stderr)
#         return []


# def exemplar_full_text(exemplar: dict) -> str:
#     """Concatenate all report fields of one retrieved exemplar's record into
#     a single text blob for n-gram / entity matching."""
#     record = exemplar.get("record", exemplar)
#     parts = []
#     for field in REPORT_FIELDS:
#         val = record.get(field, "")
#         if val and str(val).strip().lower() != "nan":
#             parts.append(str(val))
#     return " ".join(parts)


# # ----------------------------------------------------------------------
# # Per-metric computation over the whole dataframe
# # ----------------------------------------------------------------------
# def compute_all_metrics(df: pd.DataFrame, exemplar_col: str = None) -> pd.DataFrame:
#     rows = []

#     rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
#     cider_scorer = Cider()

#     # CIDEr needs the whole corpus at once (it uses corpus-level IDF), so we
#     # collect (gts, res) dicts keyed by case id first, score after the loop.
#     cider_gts, cider_res = {}, {}

#     for idx, row in df.iterrows():
#         case_id = row["Case ID"]
#         gt_raw = row["Ground Truth Report"]
#         gen_raw = row["LLM-Generated Report"]

#         gt_norm = normalize(gt_raw)
#         gen_norm = normalize(gen_raw)
#         gt_tokens = tokenize(gt_raw)
#         gen_tokens = tokenize(gen_raw)

#         # ---- BLEU-4 (sentence-level via sacrebleu, corpus BLEU computed later) ----
#         # (kept per-case for completeness / error analysis; corpus BLEU-4 for the
#         #  paper's headline number is computed once at the end over all cases)

#         # ---- ROUGE-L ----
#         rouge_l = rouge.score(gt_norm, gen_norm)["rougeL"].fmeasure

#         # ---- METEOR ----
#         try:
#             meteor = meteor_score([gt_tokens], gen_tokens)
#         except Exception:
#             meteor = np.nan

#         # ---- CIDEr bookkeeping ----
#         cider_gts[str(case_id)] = [gt_norm]
#         cider_res[str(case_id)] = [gen_norm]

#         # ---- Entity extraction ----
#         gt_teeth = extract_fdi_teeth(gt_raw)
#         gen_teeth = extract_fdi_teeth(gen_raw)
#         gt_icd = extract_icd10(gt_raw)
#         gen_icd = extract_icd10(gen_raw)

#         _, _, fdi_f1 = set_f1(gt_teeth, gen_teeth)
#         _, _, icd_f1 = set_f1(gt_icd, gen_icd)

#         # ---- Hallucination / factual-consistency rate ----
#         # entities claimed in the generated report but absent from the GT reference
#         gt_entities = gt_teeth | gt_icd
#         gen_entities = gen_teeth | gen_icd
#         if gen_entities:
#             unsupported = gen_entities - gt_entities
#             hallucination_rate = len(unsupported) / len(gen_entities)
#         else:
#             hallucination_rate = np.nan  # nothing asserted -> rate undefined

#         # ---- Retrieval-dependent metrics ----
#         retrieval_copy_rate = np.nan
#         fabricated_tooth_rate = np.nan
#         n_exemplars = 0

#         if exemplar_col is not None:
#             exemplars = parse_exemplars(row.get(exemplar_col))
#             n_exemplars = len(exemplars)
#             if exemplars:
#                 exemplar_texts = [exemplar_full_text(e) for e in exemplars]
#                 exemplar_tokens_concat = []
#                 exemplar_ngrams = set()
#                 exemplar_teeth = set()
#                 for etext in exemplar_texts:
#                     etoks = tokenize(etext)
#                     exemplar_ngrams |= get_ngrams(etoks, 4)
#                     exemplar_teeth |= extract_fdi_teeth(etext)

#                 # --- Retrieval copy rate ---
#                 gen_ngrams_list = all_ngrams_list(gen_tokens, 4)
#                 ref_ngrams = get_ngrams(gt_tokens, 4)
#                 if gen_ngrams_list:
#                     copied = [g for g in gen_ngrams_list
#                               if g in exemplar_ngrams and g not in ref_ngrams]
#                     retrieval_copy_rate = len(copied) / len(gen_ngrams_list)
#                 else:
#                     retrieval_copy_rate = np.nan

#                 # --- Fabricated tooth-reference rate ---
#                 # evidence = union of GT teeth and all retrieved exemplars' teeth
#                 evidence_teeth = gt_teeth | exemplar_teeth
#                 if gen_teeth:
#                     fabricated = gen_teeth - evidence_teeth
#                     fabricated_tooth_rate = len(fabricated) / len(gen_teeth)
#                 else:
#                     fabricated_tooth_rate = np.nan
#             else:
#                 # no exemplars retrieved/parsed for this row -> fall back to
#                 # GT-only evidence for the fabrication check; copy rate stays N/A
#                 if gen_teeth:
#                     fabricated = gen_teeth - gt_teeth
#                     fabricated_tooth_rate = len(fabricated) / len(gen_teeth)

#         rows.append({
#             "Case ID": case_id,
#             "ROUGE-L": rouge_l,
#             "METEOR": meteor,
#             "FDI_Tooth_F1": fdi_f1,
#             "ICD10_F1": icd_f1,
#             "Hallucination_Rate": hallucination_rate,
#             "Retrieval_Copy_Rate": retrieval_copy_rate,
#             "Fabricated_Tooth_Rate": fabricated_tooth_rate,
#             "N_Retrieved_Exemplars": n_exemplars,
#             "GT_Teeth": sorted(gt_teeth),
#             "Gen_Teeth": sorted(gen_teeth),
#             "GT_ICD10": sorted(gt_icd),
#             "Gen_ICD10": sorted(gen_icd),
#         })

#     per_case_df = pd.DataFrame(rows)

#     # ---- Corpus-level BLEU-4 (sacrebleu, standard for report/caption generation) ----
#     hyps = [normalize(t) for t in df["LLM-Generated Report"].tolist()]
#     refs = [[normalize(t) for t in df["Ground Truth Report"].tolist()]]
#     bleu4 = sacrebleu.corpus_bleu(hyps, refs).score / 100.0  # sacrebleu returns 0-100

#     # also compute a per-case BLEU-4 (sentence-level, smoothed) for the per-case CSV
#     sentence_bleu = []
#     for hyp, ref in zip(df["LLM-Generated Report"], df["Ground Truth Report"]):
#         score = sacrebleu.sentence_bleu(normalize(hyp), [normalize(ref)]).score / 100.0
#         sentence_bleu.append(score)
#     per_case_df.insert(1, "BLEU-4", sentence_bleu)

#     # ---- Corpus-level CIDEr ----
#     cider_score, cider_per_case = cider_scorer.compute_score(cider_gts, cider_res)
#     # cider_per_case is aligned to the dict iteration order of cider_res
#     case_order = list(cider_res.keys())
#     cider_map = dict(zip(case_order, cider_per_case))
#     per_case_df["CIDEr"] = per_case_df["Case ID"].astype(str).map(cider_map)

#     return per_case_df, bleu4, cider_score


# def summarize(per_case_df: pd.DataFrame, corpus_bleu4: float, corpus_cider: float) -> pd.DataFrame:
#     def mean_std(col):
#         vals = per_case_df[col].dropna()
#         return vals.mean(), vals.std()

#     summary = {}
#     summary["BLEU-4 (corpus)"] = (corpus_bleu4, np.nan)
#     summary["BLEU-4 (mean of per-case)"] = mean_std("BLEU-4")
#     summary["ROUGE-L"] = mean_std("ROUGE-L")
#     summary["METEOR"] = mean_std("METEOR")
#     summary["CIDEr (corpus)"] = (corpus_cider, np.nan)
#     summary["CIDEr (mean of per-case)"] = mean_std("CIDEr")
#     summary["FDI Tooth-Set F1 (macro)"] = mean_std("FDI_Tooth_F1")
#     summary["ICD-10 Set F1 (macro)"] = mean_std("ICD10_F1")
#     summary["Hallucination Rate"] = mean_std("Hallucination_Rate")
#     summary["Retrieval Copy Rate"] = mean_std("Retrieval_Copy_Rate")
#     summary["Fabricated Tooth-Reference Rate"] = mean_std("Fabricated_Tooth_Rate")

#     out = pd.DataFrame(
#         [(k, v[0], v[1]) for k, v in summary.items()],
#         columns=["Metric", "Mean", "Std"],
#     )
#     return out


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("xlsx_path", help="Path to ground_truth_vs_llm_report.xlsx")
#     parser.add_argument("--sheet", default=0, help="Sheet name or index (default: first sheet)")
#     parser.add_argument(
#         "--exemplar-col", default="Retrieved Exemplars",
#         help="Name of the column containing the JSON list of retrieved exemplars "
#              "(default: 'Retrieved Exemplars'). If this column is absent from the "
#              "sheet, retrieval-dependent metrics are skipped/marked N/A.",
#     )
#     parser.add_argument("--out-prefix", default="metrics", help="Prefix for output CSV files")
#     args = parser.parse_args()

#     df = pd.read_excel(args.xlsx_path, sheet_name=args.sheet)

#     required = {"Case ID", "Ground Truth Report", "LLM-Generated Report"}
#     missing = required - set(df.columns)
#     if missing:
#         raise ValueError(f"Input sheet is missing required column(s): {missing}")

#     exemplar_col = args.exemplar_col if args.exemplar_col in df.columns else None
#     if exemplar_col is None:
#         print(f"[info] Column '{args.exemplar_col}' not found -> "
#               f"Retrieval Copy Rate will be N/A, Fabricated Tooth Rate will fall back "
#               f"to GT-only evidence for every case.", file=sys.stderr)

#     per_case_df, corpus_bleu4, corpus_cider = compute_all_metrics(df, exemplar_col)
#     summary_df = summarize(per_case_df, corpus_bleu4, corpus_cider)

#     per_case_out = f"{args.out_prefix}_per_case.csv"
#     summary_out = f"{args.out_prefix}_summary.csv"
#     per_case_df.to_csv(per_case_out, index=False)
#     summary_df.to_csv(summary_out, index=False)

#     pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
#     print("\n================= PER-CASE METRICS (head) =================")
#     print(per_case_df.drop(columns=["GT_Teeth", "Gen_Teeth", "GT_ICD10", "Gen_ICD10"]).to_string(index=False))
#     print("\n================= SUMMARY (paste into paper) =================")
#     print(summary_df.to_string(index=False))
#     print(f"\nSaved: {per_case_out}\nSaved: {summary_out}")


# if __name__ == "__main__":
#     main()


"""
compute_metrics.py
===================
Computes evaluation metrics for the retrieval-grounded dental CBCT
report-generation paper from an Excel sheet of (Ground Truth, LLM-Generated)
report pairs over the 5 scored target fields (Oral Check, Diagnosis,
Treatment plan, Handle, Doctor advices).

METRICS
-------
1. BLEU-4, ROUGE-L, METEOR, CIDEr
2. FDI Tooth-Set F1        (tooth references anchored to 'tooth NN' / '*NN'
                             mentions only — see extract_fdi_teeth)
3. ICD-10 Set F1           (tolerant of 'K07.305' and spaced 'k 07.305')
4. Hallucination Rate      (asserted entities absent from ground truth)
5. Retrieval Copy Rate / Fabricated Tooth Rate (if exemplar column present)
6. Data-integrity checks: flags exact-duplicate generated reports across
   cases, and reports verbatim word-for-word matches to retrieved exemplars
   — both indicate fallback/copy contamination rather than genuine
   generation, and should be resolved before trusting the metrics below.

INPUT
-----
.xlsx with columns: "Case ID", "Ground Truth Report", "LLM-Generated Report"
Optional: "Retrieved Exemplars" (JSON list of {"record": {...}, ...})

USAGE
-----
    python compute_metrics.py /path/to/ground_truth_vs_llm_report.xlsx
"""

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

import sacrebleu
from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider

for pkg in ["wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


# ----------------------------------------------------------------------
# Text normalization / tokenization
# ----------------------------------------------------------------------
def normalize(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("&#39;", "'").replace("&amp;", "&").replace("&quot;", '"')
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str):
    return normalize(text).split()


def get_ngrams(tokens, n=4):
    return set(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def all_ngrams_list(tokens, n=4):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# ----------------------------------------------------------------------
# Clinical entity extraction
# ----------------------------------------------------------------------
# FIX vs. earlier version: bare \b([1-8])([1-8])\b matched incidental numbers
# ("16.5mm", "24 hours", "40 minutes"). Tooth references in this dataset are
# always anchored either to the word "tooth" or to a "*NN" shorthand used
# once a tooth has been introduced earlier in the same record. Anchoring
# extraction to these markers removes the vast majority of false positives
# from measurements/durations while still catching the two dominant real
# tooth-reference patterns in the ground truth.
TOOTH_WORD_RE = re.compile(r"\btooth\s+([1-8][1-8])\b")
TOOTH_STAR_RE = re.compile(r"(?<!\d)\*\s*([1-8][1-8])\b")

# FIX vs. earlier version: the old regex required a contiguous "K07.305" with
# no space, but the ground truth frequently writes "k 07.305" (space after
# the letter) and machine-translated variants. This version tolerates an
# optional space/hyphen after the letter. Non-ICD local-clinic shorthand like
# "lc 06" is intentionally NOT matched here (it isn't a real ICD-10 code);
# it's tracked separately by LOCAL_CODE_RE purely for visibility, and never
# folded into ICD-10 F1 / hallucination scoring.
ICD10_RE = re.compile(r"\b([A-Za-z])[\s\-]?(\d{2}\.\d{1,4}(?:x\d{3})?)\b", re.IGNORECASE)
LOCAL_CODE_RE = re.compile(r"\blc[\s\-]?\d{2,3}\b", re.IGNORECASE)


def extract_fdi_teeth(text: str) -> set:
    text = normalize(text)
    teeth = set(TOOTH_WORD_RE.findall(text)) | set(TOOTH_STAR_RE.findall(text))
    return teeth


def extract_icd10(text: str) -> set:
    text = normalize(text)
    return {f"{letter.upper()}{digits}" for letter, digits in ICD10_RE.findall(text)}


def extract_local_codes(text: str) -> set:
    text = normalize(text)
    return {m.group(0).replace(" ", "").replace("-", "").upper() for m in LOCAL_CODE_RE.finditer(text)}


# ----------------------------------------------------------------------
# Set-based F1 (macro over patients)
# ----------------------------------------------------------------------
def set_f1(ref_set: set, hyp_set: set):
    if not ref_set and not hyp_set:
        return 1.0, 1.0, 1.0
    if not hyp_set:
        return np.nan, 0.0, 0.0
    if not ref_set:
        return 0.0, np.nan, 0.0
    tp = len(ref_set & hyp_set)
    precision = tp / len(hyp_set)
    recall = tp / len(ref_set)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


# ----------------------------------------------------------------------
# Retrieved-exemplar parsing
# ----------------------------------------------------------------------
TARGET_FIELDS = ["Oral Check", "Diagnosis", "Treatment plan", "Handle", "Doctor advices"]


def parse_exemplars(cell) -> list:
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return cell
    cell = str(cell).strip()
    if not cell:
        return []
    try:
        return json.loads(cell)
    except json.JSONDecodeError:
        pass
    try:
        import ast
        return ast.literal_eval(cell)
    except Exception:
        print(f"  [warn] could not parse Retrieved Exemplars cell, skipping: {cell[:80]}...",
              file=sys.stderr)
        return []


def exemplar_full_text(exemplar: dict) -> str:
    record = exemplar.get("record", exemplar)
    parts = []
    for field in TARGET_FIELDS:
        val = record.get(field, "")
        if val and str(val).strip().lower() != "nan":
            parts.append(str(val))
    return " ".join(parts)


# ----------------------------------------------------------------------
# Data-integrity checks (run BEFORE trusting any metric below)
# ----------------------------------------------------------------------
def integrity_checks(df: pd.DataFrame, exemplar_col: str = None):
    warnings = []

    # 1. Exact-duplicate generated reports across different cases.
    gen_norm = df["LLM-Generated Report"].apply(normalize)
    dupe_mask = gen_norm.duplicated(keep=False)
    if dupe_mask.any():
        dupe_groups = df.loc[dupe_mask].groupby(gen_norm[dupe_mask]).apply(
            lambda g: list(g["Case ID"])
        )
        for ids in dupe_groups:
            if len(ids) > 1:
                warnings.append(
                    f"[INTEGRITY] Cases {ids} have byte-identical 'LLM-Generated Report' "
                    f"text. This almost always means generation failed for these cases and "
                    f"a fallback/default record was used instead of a real prediction. "
                    f"Metrics computed over these rows do not reflect model performance."
                )

    # 2. Verbatim (whole-record) copies of a retrieved exemplar.
    if exemplar_col is not None:
        for _, row in df.iterrows():
            gen_norm_text = normalize(row["LLM-Generated Report"])
            exemplars = parse_exemplars(row.get(exemplar_col))
            for ex in exemplars:
                ex_text_norm = normalize(exemplar_full_text(ex))
                if ex_text_norm and gen_norm_text == ex_text_norm:
                    warnings.append(
                        f"[INTEGRITY] Case {row['Case ID']}'s generated report is a "
                        f"word-for-word copy of retrieved exemplar case "
                        f"{ex.get('case_id', '?')}. This is not a generated report for "
                        f"this patient; exclude or re-generate before scoring."
                    )

    if warnings:
        print("\n" + "=" * 78, file=sys.stderr)
        print("DATA INTEGRITY WARNINGS — review before trusting the metrics below:", file=sys.stderr)
        for w in warnings:
            print("  " + w, file=sys.stderr)
        print("=" * 78 + "\n", file=sys.stderr)

    return warnings


# ----------------------------------------------------------------------
# Per-metric computation over the whole dataframe
# ----------------------------------------------------------------------
def compute_all_metrics(df: pd.DataFrame, exemplar_col: str = None) -> pd.DataFrame:
    rows = []
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    cider_scorer = Cider()
    cider_gts, cider_res = {}, {}

    for idx, row in df.iterrows():
        case_id = row["Case ID"]
        gt_raw = row["Ground Truth Report"]
        gen_raw = row["LLM-Generated Report"]

        gt_norm = normalize(gt_raw)
        gen_norm = normalize(gen_raw)
        gt_tokens = tokenize(gt_raw)
        gen_tokens = tokenize(gen_raw)

        rouge_l = rouge.score(gt_norm, gen_norm)["rougeL"].fmeasure

        try:
            meteor = meteor_score([gt_tokens], gen_tokens)
        except Exception:
            meteor = np.nan

        cider_gts[str(case_id)] = [gt_norm]
        cider_res[str(case_id)] = [gen_norm]

        gt_teeth = extract_fdi_teeth(gt_raw)
        gen_teeth = extract_fdi_teeth(gen_raw)
        gt_icd = extract_icd10(gt_raw)
        gen_icd = extract_icd10(gen_raw)
        gt_local = extract_local_codes(gt_raw)
        gen_local = extract_local_codes(gen_raw)

        _, _, fdi_f1 = set_f1(gt_teeth, gen_teeth)
        _, _, icd_f1 = set_f1(gt_icd, gen_icd)

        gt_entities = gt_teeth | gt_icd
        gen_entities = gen_teeth | gen_icd
        if gen_entities:
            unsupported = gen_entities - gt_entities
            hallucination_rate = len(unsupported) / len(gen_entities)
        else:
            hallucination_rate = np.nan

        retrieval_copy_rate = np.nan
        fabricated_tooth_rate = np.nan
        n_exemplars = 0

        if exemplar_col is not None:
            exemplars = parse_exemplars(row.get(exemplar_col))
            n_exemplars = len(exemplars)
            if exemplars:
                exemplar_texts = [exemplar_full_text(e) for e in exemplars]
                exemplar_ngrams = set()
                exemplar_teeth = set()
                for etext in exemplar_texts:
                    etoks = tokenize(etext)
                    exemplar_ngrams |= get_ngrams(etoks, 4)
                    exemplar_teeth |= extract_fdi_teeth(etext)

                gen_ngrams_list = all_ngrams_list(gen_tokens, 4)
                ref_ngrams = get_ngrams(gt_tokens, 4)
                if gen_ngrams_list:
                    copied = [g for g in gen_ngrams_list
                              if g in exemplar_ngrams and g not in ref_ngrams]
                    retrieval_copy_rate = len(copied) / len(gen_ngrams_list)

                evidence_teeth = gt_teeth | exemplar_teeth
                if gen_teeth:
                    fabricated = gen_teeth - evidence_teeth
                    fabricated_tooth_rate = len(fabricated) / len(gen_teeth)
            else:
                if gen_teeth:
                    fabricated = gen_teeth - gt_teeth
                    fabricated_tooth_rate = len(fabricated) / len(gen_teeth)

        rows.append({
            "Case ID": case_id,
            "ROUGE-L": rouge_l,
            "METEOR": meteor,
            "FDI_Tooth_F1": fdi_f1,
            "ICD10_F1": icd_f1,
            "Hallucination_Rate": hallucination_rate,
            "Retrieval_Copy_Rate": retrieval_copy_rate,
            "Fabricated_Tooth_Rate": fabricated_tooth_rate,
            "N_Retrieved_Exemplars": n_exemplars,
            "GT_Teeth": sorted(gt_teeth),
            "Gen_Teeth": sorted(gen_teeth),
            "GT_ICD10": sorted(gt_icd),
            "Gen_ICD10": sorted(gen_icd),
            "GT_LocalCodes": sorted(gt_local),
            "Gen_LocalCodes": sorted(gen_local),
        })

    per_case_df = pd.DataFrame(rows)

    hyps = [normalize(t) for t in df["LLM-Generated Report"].tolist()]
    refs = [[normalize(t) for t in df["Ground Truth Report"].tolist()]]
    bleu4 = sacrebleu.corpus_bleu(hyps, refs).score / 100.0

    sentence_bleu = []
    for hyp, ref in zip(df["LLM-Generated Report"], df["Ground Truth Report"]):
        score = sacrebleu.sentence_bleu(normalize(hyp), [normalize(ref)]).score / 100.0
        sentence_bleu.append(score)
    per_case_df.insert(1, "BLEU-4", sentence_bleu)

    cider_score, cider_per_case = cider_scorer.compute_score(cider_gts, cider_res)
    case_order = list(cider_res.keys())
    cider_map = dict(zip(case_order, cider_per_case))
    per_case_df["CIDEr"] = per_case_df["Case ID"].astype(str).map(cider_map)

    return per_case_df, bleu4, cider_score


def summarize(per_case_df: pd.DataFrame, corpus_bleu4: float, corpus_cider: float) -> pd.DataFrame:
    def mean_std(col):
        vals = per_case_df[col].dropna()
        return vals.mean(), vals.std()

    summary = {
        "BLEU-4 (corpus)": (corpus_bleu4, np.nan),
        "BLEU-4 (mean of per-case)": mean_std("BLEU-4"),
        "ROUGE-L": mean_std("ROUGE-L"),
        "METEOR": mean_std("METEOR"),
        "CIDEr (corpus)": (corpus_cider, np.nan),
        "CIDEr (mean of per-case)": mean_std("CIDEr"),
        "FDI Tooth-Set F1 (macro)": mean_std("FDI_Tooth_F1"),
        "ICD-10 Set F1 (macro)": mean_std("ICD10_F1"),
        "Hallucination Rate": mean_std("Hallucination_Rate"),
        "Retrieval Copy Rate": mean_std("Retrieval_Copy_Rate"),
        "Fabricated Tooth-Reference Rate": mean_std("Fabricated_Tooth_Rate"),
    }
    return pd.DataFrame(
        [(k, v[0], v[1]) for k, v in summary.items()],
        columns=["Metric", "Mean", "Std"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xlsx_path")
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--exemplar-col", default="Retrieved Exemplars")
    parser.add_argument("--out-prefix", default="metrics")
    args = parser.parse_args()

    df = pd.read_excel(args.xlsx_path, sheet_name=args.sheet)

    required = {"Case ID", "Ground Truth Report", "LLM-Generated Report"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input sheet is missing required column(s): {missing}")

    exemplar_col = args.exemplar_col if args.exemplar_col in df.columns else None
    if exemplar_col is None:
        print(f"[info] Column '{args.exemplar_col}' not found -> "
              f"Retrieval Copy Rate will be N/A, Fabricated Tooth Rate will fall back "
              f"to GT-only evidence for every case.", file=sys.stderr)

    integrity_checks(df, exemplar_col)

    per_case_df, corpus_bleu4, corpus_cider = compute_all_metrics(df, exemplar_col)
    summary_df = summarize(per_case_df, corpus_bleu4, corpus_cider)

    per_case_out = f"{args.out_prefix}_per_case.csv"
    summary_out = f"{args.out_prefix}_summary.csv"
    per_case_df.to_csv(per_case_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    pd.set_option("display.float_format", lambda x: f"{x:0.4f}")
    drop_cols = ["GT_Teeth", "Gen_Teeth", "GT_ICD10", "Gen_ICD10", "GT_LocalCodes", "Gen_LocalCodes"]
    print("\n================= PER-CASE METRICS (head) =================")
    print(per_case_df.drop(columns=drop_cols).to_string(index=False))
    print("\n================= SUMMARY (paste into paper) =================")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {per_case_out}\nSaved: {summary_out}")


if __name__ == "__main__":
    main()
