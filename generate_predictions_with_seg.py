"""
generate_predictions_multimodal.py

Inference with the LoRA-fine-tuned MedGemma 1.5 4B over the (unchanged) RAG output,
with the extra inputs from the "Image and Text Inputs to Improve LLM Report Generation"
idea:

  --mode text          system 1: age/sex/main appeal + 5 retrieved records
  --mode facts         system 2: + tooth facts file derived from the U-Mamba2 mask
  --mode facts_images  system 3: + overview slices with mask outline + tooth-centred crops

Per patient it writes
  * predictions.json   clean 3-field assessment (source tags stripped) -> used for scoring
  * reports.json       tagged fields, validation flags, mismatch flag and the 7-section draft
  * generation_status.csv, submission.zip, (optional) overlay PNGs for clinician QC

Expected inputs (all keyed by case id):
  --volume_dir/<case_id>.nii.gz   CBCT volume
  --mask_dir/<case_id>.nii.gz     U-Mamba2 tooth mask on the scan's own grid
  --slice_index_json              {case_id: [20 absolute axial indices used for BiomedCLIP]}
  --facts_dir/<case_id>.txt       optional cache of the facts file (computed if absent)
"""

import os
import re
import csv
import json
import html
import argparse
import zipfile
import traceback
from pathlib import Path

import numpy as np
import torch
import nibabel as nib
from PIL import Image
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# Target = the patient's own Oral Check / Diagnosis / Treatment plan (idea doc, sec. 5).
# Handle and Doctor advices are no longer generated. Add them back here if you still score them.
TARGET_FIELDS = ["Oral Check", "Diagnosis", "Treatment plan"]
UNABLE = "Unable to evaluate"

QUADRANT = {1: "upper right", 2: "upper left", 3: "lower left", 4: "lower right"}
POSITION = {1: "central incisor", 2: "lateral incisor", 3: "canine", 4: "first premolar",
            5: "second premolar", 6: "first molar", 7: "second molar", 8: "third molar"}
ALL_FDI = [q * 10 + n for q in range(1, 5) for n in range(1, 9)]

# Structure label ids of YOUR trained segmentation model. Teeth default to label == FDI number.
# Override with --label_map JSON, e.g.
#   {"structures": {"canal_right": 103, "canal_left": 104, "sinus_right": 105, "sinus_left": 106},
#    "teeth": {"11": 1, "12": 2}}          # fdi -> label, only if labels are not FDI numbers
DEFAULT_STRUCTURES = {"canal_right": None, "canal_left": None, "sinus_right": None, "sinus_left": None}

# illustrative; align with your dataset's "Diagnosis categories" vocabulary
CATEGORY_KEYWORDS = {
    "Caries": ["caries", "carious"],
    "Pulpitis": ["pulpitis"],
    "Periodontitis": ["periodont"],
    "Periapical disease": ["periapical", "apical"],
    "Missing Teeth": ["missing", "edentul"],
    "Impacted Teeth": ["impact"],
    "Malocclusion": ["crowd", "malocclusion", "malposition"],
}

TAG_RE = re.compile(
    r"\[\s*((?:patient|seg|case\s*[A-Z])(?:\s*,\s*(?:patient|seg|case\s*[A-Z]))*)\s*\]", re.IGNORECASE
)
TOOTH_LIST_RE = re.compile(
    r"(?:tooth|teeth|#|fdi)\s*#?\s*(\d{2}(?:\s*(?:,|and|&|/|-)\s*\d{2})*)", re.IGNORECASE
)
ICD_RE = re.compile(r"\b[A-Z]\d{2}(?:\.\d+)?\b")


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
def build_system_prompt(mode):
    has_seg = mode in ("facts", "facts_images")
    has_img = mode == "facts_images"

    sources = (
        "SOURCE TAGS - every fragment you write must end with the tag(s) of what supports it:\n"
        "  [patient]  the CURRENT patient's age, sex, main appeal\n"
        "  [case A] .. [case E]  a reference record of ANOTHER patient\n"
    )
    inputs = "- the CURRENT patient's age, sex and main appeal\n"
    if has_seg:
        sources += "  [seg]  the segmentation facts file (automatic tooth segmentation of THIS patient's CBCT)\n"
        inputs += (
            "- a segmentation facts file: which numbered teeth the automatic model found, where, "
            "their size, distance to canal/sinus, and a complaint/mask mismatch flag\n"
        )
    if has_img:
        inputs += (
            "- CBCT slices of THIS patient: overview axial slices and tooth-centred crops. "
            "A thin red outline marks segmented teeth; yellow marks the tooth named in the complaint\n"
        )
    inputs += "- reference records of other patients with visually similar CBCT scans\n"

    rules = (
        "RULES:\n"
        "- Reference records are OTHER patients. Reuse their clinical facts (FDI numbers, ICD-10 codes, "
        "usual procedures) only when they fit THIS patient's complaint"
        + (" and segmentation facts" if has_seg else "")
        + ". Never write a reference record's finding as this patient's own finding. "
        "Never copy 4 or more consecutive words from a reference record; write fresh, telegraphic fragments.\n"
    )
    if has_seg:
        rules += (
            "- 'Not detected' in the facts file is NOT 'missing': it can be impaction, a segmentation "
            "failure or a cropped field of view. Never call a tooth missing on that basis alone.\n"
            "- Only cite tooth numbers that appear in the facts file's detected list or in the main appeal. "
            "If the complaint names a tooth the mask did not detect, say it cannot be evaluated.\n"
        )
    if has_img:
        rules += (
            "- You are not asked to read lesions from the images. Use them for anatomy and context only. "
            "If an item cannot be judged from the evidence, write 'unable to evaluate'.\n"
        )
    else:
        rules += "- If an item cannot be judged from the evidence, write 'unable to evaluate'.\n"
    rules += (
        "- Tooth format: 'tooth NN (anatomical position)', e.g. 'tooth 36 (lower left first molar)'. Never '#NN'.\n"
        "- ICD-10 codes only if they appear in a reference record; format 'condition (ICD-10)'.\n"
        "- Diagnosis and Treatment plan must be consistent with the Oral Check you write.\n"
        "- Short telegraphic fragments; cover all relevant teeth/conditions; no padding.\n"
        "- OUTPUT: a single JSON object with exactly these keys in order: "
        + ", ".join(TARGET_FIELDS)
        + ". Each value is a string. No markdown, nothing before or after the JSON."
    )

    return (
        "You are an expert dental clinician drafting the post-examination assessment for the CURRENT "
        "patient. This is a DRAFT for clinician review.\n\n"
        "You are given:\n" + inputs + "\n" + sources + "\n" + rules
    )


RETRY_NOTE = (
    "\n\nIMPORTANT: your previous draft copied a reference record almost word-for-word or cited teeth/"
    "findings not supported by this patient's evidence. Discard it. Write a new assessment grounded in "
    "THIS patient's own evidence, using reference records for terminology only."
)


def is_placeholder(v):
    return str(v).strip().lower() in ("", "not available", "unknown", "none", "n/a", "nan", "null")


def clean_text(v):
    return html.unescape(v).strip() if isinstance(v, str) else v


def parse_query_text(query_text):
    info = {"age": "unknown", "sex": "unknown", "main_appeal": None}
    if not query_text:
        return info
    m = re.search(r"\bAge:\s*([^.\n]+)", query_text, re.I)
    if m:
        info["age"] = m.group(1).strip()
    m = re.search(r"\bSex:\s*([^.\n]+)", query_text, re.I)
    if m:
        info["sex"] = m.group(1).strip()
    m = re.search(r"\bMain appeal:\s*(.+)$", query_text, re.I | re.S)
    if m:
        info["main_appeal"] = m.group(1).strip()
    return info


def format_query_context(case):
    pc = case.get("patient_context", {}) or {}
    qt = parse_query_text(case.get("query_text", ""))
    age = pc.get("age") or qt["age"]
    age_group = pc.get("age_group", "")
    sex = pc.get("sex") or qt["sex"]
    appeal = (
        case.get("main_appeal") or case.get("Main appeal") or pc.get("main_appeal")
        or pc.get("Main appeal") or case.get("chief_complaint") or pc.get("chief_complaint")
        or qt["main_appeal"] or "Not recorded"
    )
    return {
        "Age": f"{age} ({age_group})" if age_group else str(age),
        "Sex": str(sex),
        "Main appeal": clean_text(str(appeal)),
    }


def format_exemplars(case, max_exemplars, max_chars=400):
    """Sec. 6: Case ID, age group, sex, main appeal, oral check, diagnosis, categories, plan, rank, score.
    Subsequent/medical history, Handle and Doctor advices are NOT passed."""
    fields = ["Age group", "Sex", "Main appeal", "Oral Check", "Diagnosis",
              "Diagnosis categories", "Treatment plan"]
    blocks = []
    for i, ex in enumerate((case.get("retrieved_exemplars") or [])[:max_exemplars]):
        if not isinstance(ex, dict):
            continue
        rec = ex.get("record", {}) or {}
        score = ex.get("similarity") or ex.get("cosine") or ex.get("reranker_score") or 0.0
        rank = ex.get("rank", i + 1)
        lines = [f"[case {chr(65 + i)}] (retrieval rank {rank}, cosine {float(score):.3f}, "
                 f"id {rec.get('Case ID', ex.get('case_id', '?'))})"]
        for f in fields:
            val = clean_text(str(rec.get(f, "Not available")))
            lines.append(f"  {f}: {val[:max_chars] + '...' if len(val) > max_chars else val}")
        blocks.append("\n".join(lines))
    return ("REFERENCE RECORDS (OTHER patients with similar CBCT scans; not this patient):\n\n"
            + "\n\n".join(blocks))


def build_messages(case, mode, facts_text, images, max_exemplars, correction=False):
    """NOTE: reuse this exact function (or mirror it) when building the LoRA training prompts."""
    q = format_query_context(case)
    text = (
        "CURRENT PATIENT [patient]:\n"
        f"  Age: {q['Age']}\n  Sex: {q['Sex']}\n  Main appeal: {q['Main appeal']}"
    )
    if mode in ("facts", "facts_images"):
        text += "\n\nSEGMENTATION FACTS [seg]:\n" + (facts_text or "Not available")

    user = [{"type": "text", "text": text}]
    if mode == "facts_images" and images:
        user.append({"type": "text", "text": "\nCBCT IMAGES of the CURRENT patient:"})
        for caption, img in images:
            user.append({"type": "text", "text": caption})
            user.append({"type": "image", "image": img})
    tail = (
        "\n\n" + format_exemplars(case, max_exemplars)
        + "\n\nWrite the JSON assessment for the CURRENT patient with keys: "
        + ", ".join(TARGET_FIELDS) + ". End every fragment with its source tag(s)."
    )
    if correction:
        tail += RETRY_NOTE
    user.append({"type": "text", "text": tail})

    return [
        {"role": "system", "content": [{"type": "text", "text": build_system_prompt(mode)}]},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- #
# Tooth helpers
# --------------------------------------------------------------------------- #
def is_fdi(n):
    n = int(n)
    return 1 <= n // 10 <= 4 and 1 <= n % 10 <= 8


def tooth_name(fdi):
    fdi = int(fdi)
    return f"{QUADRANT[fdi // 10]} {POSITION[fdi % 10]}"


def extract_teeth(text):
    out = set()
    for m in TOOTH_LIST_RE.finditer(text or ""):
        for n in re.findall(r"\d{2}", m.group(1)):
            if is_fdi(n):
                out.add(int(n))
    return out


def normalize_tooth_notation(text):
    if not isinstance(text, str):
        return text
    text = clean_text(text)
    text = re.sub(r"tooth\s*#\s*(\d{1,2})", r"tooth \1", text, flags=re.I)
    return re.sub(r"#(\d{1,2})\b", r"tooth \1", text)


def add_tooth_names(text):
    def repl(m):
        n = m.group(1)
        return f"tooth {n} ({tooth_name(n)})" if is_fdi(n) else m.group(0)
    return re.sub(r"\btooth\s*(\d{2})\b(?!\s*\()", repl, text, flags=re.I)


# --------------------------------------------------------------------------- #
# Volume / mask loading and facts file
# --------------------------------------------------------------------------- #
def find_file(directory, case_id):
    for ext in (".nii.gz", ".nii"):
        p = Path(directory) / f"{case_id}{ext}"
        if p.exists():
            return p
    return None


def load_nifti(path, slice_axis):
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    order = [a for a in range(3) if a != slice_axis] + [slice_axis]  # slices -> last axis
    return np.transpose(data, order), spacing[order]


def compute_facts(mask, spacing, label_map, complaint_teeth, sampled_idx, min_voxels=100):
    tooth_label = {f: int(label_map.get("teeth", {}).get(str(f), f)) for f in ALL_FDI}
    structs = {**DEFAULT_STRUCTURES, **label_map.get("structures", {})}
    mask = mask.astype(np.int32)
    objs = ndi.find_objects(mask)

    def points(label):
        if label is None or label < 1 or label > len(objs) or objs[label - 1] is None:
            return None
        sl = objs[label - 1]
        sub = mask[sl] == label
        cc, n = ndi.label(sub)
        if n == 0:
            return None
        sizes = np.bincount(cc.ravel(), minlength=n + 1)[1:]
        keep = np.flatnonzero(sizes >= min_voxels) + 1
        if keep.size == 0:
            return None
        sub = np.isin(cc, keep)
        return np.argwhere(sub) + np.array([s.start for s in sl])

    struct_trees = {}
    for name, lab in structs.items():
        pts = points(lab)
        if pts is not None and len(pts):
            step = max(1, len(pts) // 20000)
            struct_trees[name] = cKDTree(pts[::step] * spacing)

    teeth, detected, not_detected = {}, [], []
    for fdi in ALL_FDI:
        pts = points(tooth_label[fdi])
        if pts is None or len(pts) == 0:
            not_detected.append(fdi)
            continue
        detected.append(fdi)
        quadrant, pos = fdi // 10, fdi % 10
        side = "right" if quadrant in (1, 4) else "left"
        jaw = "upper" if quadrant in (1, 2) else "lower"
        struct_key = ("canal_" if jaw == "lower" else "sinus_") + side
        dist = None
        if struct_key in struct_trees:
            step = max(1, len(pts) // 5000)
            dist = float(struct_trees[struct_key].query(pts[::step] * spacing)[0].min())
        z = np.unique(pts[:, 2])
        teeth[fdi] = {
            "centroid_mm": (pts.mean(0) * spacing).round(1).tolist(),
            "centroid_vox": pts.mean(0).tolist(),
            "volume_mm3": float(len(pts) * np.prod(spacing)),
            "region": "anterior" if pos <= 3 else "posterior",
            "jaw": jaw, "side": side,
            "dist_struct": ("mandibular canal" if jaw == "lower" else "maxillary sinus", dist),
            "slice_range": (int(z.min()), int(z.max())),
            "sampled_slices": [int(i) for i in sampled_idx if i in set(z.tolist())],
        }

    upper = sum(1 for f in detected if f // 10 in (1, 2))
    lower = len(detected) - upper
    mismatch = sorted(t for t in complaint_teeth if t in not_detected)

    lines = [
        "Automatic segmentation (may be wrong). 'Not detected' is NOT 'missing': it may be impaction, "
        "a segmentation failure or a cropped field of view.",
        f"Teeth detected: upper {upper}/16, lower {lower}/16.",
        "Detected teeth (centroid in mm, volume, distance to nearest canal/sinus, axial slices):",
    ]
    for fdi in detected:
        t = teeth[fdi]
        sname, sd = t["dist_struct"]
        d = f"{sd:.1f} mm to {sname}" if sd is not None else "distance n/a"
        lines.append(
            f"- tooth {fdi} ({tooth_name(fdi)}): {t['region']}, centroid {tuple(t['centroid_mm'])} mm, "
            f"{t['volume_mm3']:.0f} mm3, {d}, slices {t['slice_range'][0]}-{t['slice_range'][1]}"
        )
    lines.append("Not detected: " + (", ".join(map(str, not_detected)) if not_detected else "none"))
    if complaint_teeth:
        if mismatch:
            lines.append("MISMATCH FLAG: complaint names "
                         + ", ".join(f"tooth {t} ({tooth_name(t)})" for t in mismatch)
                         + " but the mask did not detect it.")
        else:
            lines.append("Mismatch flag: none (complaint teeth were detected).")
    return {"text": "\n".join(lines), "teeth": teeth, "detected": set(detected),
            "not_detected": not_detected, "mismatch": mismatch, "upper": upper, "lower": lower}


# --------------------------------------------------------------------------- #
# Images: same slices as BiomedCLIP, mask outline, tooth-centred crops
# --------------------------------------------------------------------------- #
def norm_slice(sl):
    """Per-slice min-max scaling. Replace with the exact scaling used in your BiomedCLIP pipeline."""
    sl = sl.astype(np.float32)
    lo, hi = float(sl.min()), float(sl.max())
    if hi <= lo:
        return np.zeros(sl.shape, np.uint8)
    return ((sl - lo) / (hi - lo) * 255).astype(np.uint8)


def draw_outline(gray, labels, highlight=None):
    rgb = np.stack([gray] * 3, -1)
    fg = labels > 0
    edge = fg & ((ndi.minimum_filter(labels, 3) != labels) | (ndi.maximum_filter(labels, 3) != labels))
    rgb[edge] = (255, 60, 60)
    if highlight is not None:
        rgb[edge & (labels == highlight)] = (255, 220, 0)
    return rgb


def render(gray, labels, out_size, highlight=None):
    """Image: bilinear/Lanczos. Mask: nearest-neighbour so labels are not blurred."""
    g = np.array(Image.fromarray(gray).resize((out_size, out_size), Image.LANCZOS))
    lab = np.array(Image.fromarray(labels.astype(np.int32)).resize((out_size, out_size), Image.NEAREST))
    return Image.fromarray(draw_outline(g, lab, highlight))


def build_images(vol, mask, facts, sampled_idx, complaint_teeth, label_map, args):
    images = []
    tooth_label = lambda f: int(label_map.get("teeth", {}).get(str(f), f))
    # a) 4 evenly spaced slices out of the 20 (same indices as BiomedCLIP)
    for pos in np.linspace(0, len(sampled_idx) - 1, 4).round().astype(int):
        k = int(sampled_idx[pos])
        img = render(norm_slice(vol[:, :, k]), mask[:, :, k], args.overview_size)
        images.append((f"Overview axial slice {k} (red outline = segmented teeth):", img))

    # b) up to 3 crops around teeth named in the complaint, cut from the full-resolution slice that intersects the tooth
    named = [t for t in sorted(complaint_teeth) if t in facts["teeth"]][:3]
    for fdi in named:
        ft = facts["teeth"][fdi]
        ci, cj, ck = ft["centroid_vox"]
        z0, z1 = ft["slice_range"]
        in_range = [s for s in sampled_idx if z0 <= s <= z1]
        k = int(min(in_range, key=lambda s: abs(s - ck))) if in_range else int(ck)
        h = args.crop_half
        i0, j0 = int(max(0, min(vol.shape[0] - 2 * h, ci - h))), int(max(0, min(vol.shape[1] - 2 * h, cj - h)))
        g = norm_slice(vol[:, :, k])[i0:i0 + 2 * h, j0:j0 + 2 * h]
        lab = mask[:, :, k][i0:i0 + 2 * h, j0:j0 + 2 * h]
        img = render(g, lab, args.crop_size, highlight=tooth_label(fdi))
        images.append((f"Crop around tooth {fdi} ({tooth_name(fdi)}), axial slice {k} "
                       f"(yellow outline = this tooth):", img))
    return images


def prepare_case_inputs(case_id, case, args, label_map, slice_index):
    """Returns dict(facts, facts_text, images, notes). Degrades gracefully if files are missing."""
    out = {"facts": None, "facts_text": None, "images": [], "notes": []}
    if args.mode == "text":
        return out

    q = format_query_context(case)
    complaint_teeth = extract_teeth(q["Main appeal"])
    out["complaint_teeth"] = complaint_teeth

    mask_path = find_file(args.mask_dir, case_id) if args.mask_dir else None
    sampled = slice_index.get(case_id)
    if mask_path is None:
        out["notes"].append("mask file not found; segmentation facts unavailable")
        out["facts_text"] = "Segmentation unavailable for this patient."
        return out

    mask, spacing = load_nifti(mask_path, args.slice_axis)
    sampled = sampled or []
    facts = compute_facts(mask, spacing, label_map, complaint_teeth, sampled,
                          args.min_component_voxels)

    cached = Path(args.facts_dir) / f"{case_id}.txt" if args.facts_dir else None
    if (cached is not None and cached.exists()
            and cached.stat().st_mtime_ns >= mask_path.stat().st_mtime_ns):
        facts["text"] = cached.read_text(encoding="utf-8")
    elif cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(facts["text"], encoding="utf-8")
    out["facts"], out["facts_text"] = facts, facts["text"]

    if args.mode == "facts_images":
        vol_path = find_file(args.volume_dir, case_id) if args.volume_dir else None
        if vol_path is None or not sampled:
            out["notes"].append("volume or slice indices missing; images skipped")
        else:
            vol, _ = load_nifti(vol_path, args.slice_axis)
            if vol.shape != mask.shape:
                raise ValueError(f"volume {vol.shape} and mask {mask.shape} shapes differ for case {case_id}")
            out["images"] = build_images(vol, mask, facts, sampled, complaint_teeth, label_map, args)
            if args.save_images_dir:
                d = Path(args.save_images_dir) / str(case_id)
                d.mkdir(parents=True, exist_ok=True)
                for n, (cap, img) in enumerate(out["images"]):
                    img.save(d / f"{n:02d}.png")
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def load_model(base_model, adapter):
    processor = AutoProcessor.from_pretrained(adapter if adapter else base_model)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


@torch.inference_mode()
def run_generation(model, processor, messages, max_new_tokens, sample, num_beams):
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    n_in = inputs["input_ids"].shape[-1]
    # NOTE: repetition_penalty / no_repeat_ngram_size from the old script are intentionally gone:
    # they penalise repeated source tags ("[patient]"), tooth formats and ICD codes.
    kw = dict(max_new_tokens=max_new_tokens)
    if sample:
        kw.update(do_sample=True, temperature=0.7, top_p=0.9)
    else:
        kw.update(do_sample=False, num_beams=num_beams)
    out = model.generate(**inputs, **kw)
    return processor.decode(out[0][n_in:], skip_special_tokens=True)


def extract_json(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {}
    cand = m.group(0)
    for c in (cand, re.sub(r",\s*([\}\]])", r"\1", cand)):
        try:
            obj = json.loads(c)
            return obj if isinstance(obj, dict) else {}
        except json.JSONDecodeError:
            continue
    return {}


# --------------------------------------------------------------------------- #
# Validation / post-processing (idea doc, sec. 7)
# --------------------------------------------------------------------------- #
def get_4grams(text):
    toks = re.sub(r"\s+", " ", str(text).strip().lower()).split()
    return {tuple(toks[i:i + 4]) for i in range(len(toks) - 3)}


def record_copy_ratio(record, exemplars):
    gen = get_4grams(" ".join(str(record.get(f, "")) for f in TARGET_FIELDS))
    if not gen:
        return 0.0
    best = 0.0
    for ex in exemplars:
        rec = ex.get("record", {}) or {}
        ng = get_4grams(" ".join(str(rec.get(f, "")) for f in TARGET_FIELDS))
        if ng:
            best = max(best, len(gen & ng) / len(gen))
    return best


def strip_tags(text):
    return re.sub(r"\s{2,}", " ", TAG_RE.sub("", text)).strip(" ;")


def allowed_teeth(case, mode, inputs, max_exemplars):
    complaint = extract_teeth(format_query_context(case)["Main appeal"])
    if mode == "text":
        ref = set()
        for ex in (case.get("retrieved_exemplars") or [])[:max_exemplars]:
            rec = ex.get("record", {}) or {}
            ref |= extract_teeth(" ".join(str(rec.get(f, "")) for f in TARGET_FIELDS))
        return complaint | ref
    facts = inputs.get("facts")
    return complaint | (facts["detected"] if facts else set())


def supported_codes(case, max_exemplars):
    codes = set()
    for ex in (case.get("retrieved_exemplars") or [])[:max_exemplars]:
        rec = ex.get("record", {}) or {}
        codes |= set(ICD_RE.findall(str(rec.get("Diagnosis", ""))))
    return codes


def postprocess(record, case, mode, inputs, max_exemplars):
    """Returns (tagged_record, clean_record, flags). Drops unsupported tooth numbers and
    similar-case findings written as the patient's own Oral Check; flags untagged/unsupported diagnoses."""
    allowed = allowed_teeth(case, mode, inputs, max_exemplars)
    codes_ok = supported_codes(case, max_exemplars)
    tagged, clean, flags = {}, {}, []

    for field in TARGET_FIELDS:
        raw = record.get(field, "")
        if isinstance(raw, list):
            raw = "; ".join(map(str, raw))
        val = normalize_tooth_notation(str(raw or "").strip())
        kept = []
        for frag in re.split(r"(?<=[.;])\s+(?=[A-Za-z\[])|\n+", val):
            frag = frag.strip()
            if not frag or is_placeholder(frag):
                continue
            bad = sorted(t for t in extract_teeth(frag) if t not in allowed)
            if bad:
                flags.append(f"{field}: removed fragment citing tooth {bad} (not in facts/complaint)")
                continue
            tags = [t.strip().lower() for m in TAG_RE.findall(frag) for t in m.split(",")]
            if not tags:
                flags.append(f"{field}: untagged fragment kept - '{frag[:60]}'")
            elif field == "Oral Check" and all(t.startswith("case") for t in tags):
                flags.append(f"{field}: removed similar-case finding presented as patient's own - '{frag[:60]}'")
                continue
            elif all(t.startswith("case") for t in tags):
                flags.append(f"{field}: supported only by a similar case - verify - '{frag[:60]}'")
            if field == "Diagnosis":
                for c in ICD_RE.findall(frag):
                    if c not in codes_ok:
                        flags.append(f"Diagnosis: ICD-10 {c} not found in any retrieved record")
                if tags and not any(t == "seg" or t == "patient" or t.startswith("case") for t in tags):
                    flags.append("Diagnosis: no valid source tag")
            kept.append(add_tooth_names(frag))
        tagged[field] = " ".join(kept) if kept else UNABLE
        clean[field] = strip_tags(tagged[field]) if kept else UNABLE
        if not clean[field]:
            clean[field] = UNABLE
    return tagged, clean, flags


def has_content(clean):
    return any(v != UNABLE for v in clean.values())


def top_case_fallback(case):
    exs = case.get("retrieved_exemplars") or []
    rec = (exs[0].get("record", {}) if exs else {}) or {}
    out = {}
    for f in TARGET_FIELDS:
        v = add_tooth_names(normalize_tooth_notation(str(rec.get(f, "")).strip()))
        out[f] = v if v and not is_placeholder(v) else UNABLE
    return out


# --------------------------------------------------------------------------- #
# Generation with one retry
# --------------------------------------------------------------------------- #
def generate_record(model, processor, case, inputs, args):
    exemplars = (case.get("retrieved_exemplars") or [])[: args.max_exemplars]

    def attempt(correction, sample):
        msgs = build_messages(case, args.mode, inputs["facts_text"], inputs["images"],
                              args.max_exemplars, correction=correction)
        raw = run_generation(model, processor, msgs, args.max_new_tokens, sample, args.num_beams)
        rec = extract_json(raw)
        if not rec:
            return None
        tagged, clean, flags = postprocess(rec, case, args.mode, inputs, args.max_exemplars)
        if not has_content(clean) or record_copy_ratio(clean, exemplars) >= args.copy_threshold:
            return None
        return tagged, clean, flags

    res = attempt(correction=False, sample=False)
    if res:
        return (*res, "model")
    res = attempt(correction=True, sample=True)
    if res:
        return (*res, "model_after_retry")

    if args.fallback == "top_case":
        clean = top_case_fallback(case)
        return clean, clean, ["fallback: copied top similar case (not patient-specific)"], "fallback"
    clean = {f: UNABLE for f in TARGET_FIELDS}
    return clean, clean, ["fallback: no valid model output"], "fallback"


# --------------------------------------------------------------------------- #
# 7-section draft report (assembled in code from validated pieces)
# --------------------------------------------------------------------------- #
def diagnosis_categories(text):
    low = text.lower()
    return [c for c, kws in CATEGORY_KEYWORDS.items() if any(k in low for k in kws)]


def build_report(case_id, case, tagged, flags, inputs, status, args):
    q = format_query_context(case)
    facts = inputs.get("facts")
    anatomy = "Not used in this configuration."
    if facts:
        anatomy = (
            f"Teeth detected by automatic segmentation: upper {facts['upper']}/16, lower {facts['lower']}/16. "
            + ("Not detected (may be impaction, segmentation failure or cropped field of view; "
               "not necessarily missing): " + ", ".join(map(str, facts["not_detected"])) + "."
               if facts["not_detected"] else "All 32 teeth detected.")
            + " [seg]"
        )
    similar = []
    for i, ex in enumerate((case.get("retrieved_exemplars") or [])[: args.max_exemplars]):
        rec = ex.get("record", {}) or {}
        score = ex.get("similarity") or ex.get("cosine") or ex.get("reranker_score") or 0.0
        similar.append(f"[case {chr(65 + i)}] id {rec.get('Case ID', '?')}, cosine {float(score):.3f}: "
                       f"{str(rec.get('Diagnosis', 'n/a'))[:160]}")

    verify = list(flags)
    if facts and facts["mismatch"]:
        verify.insert(0, "Complaint names tooth "
                      + ", ".join(map(str, facts["mismatch"])) + " that the mask did not detect.")
    for note in inputs.get("notes", []):
        verify.append(note)
    verify.append("Automatic masks can be wrong; the model does not read lesions from the images.")

    cats = diagnosis_categories(tagged["Diagnosis"])
    parts = [
        "DRAFT FOR CLINICIAN REVIEW",
        f"1. Patient summary: age {q['Age']}, sex {q['Sex']}, main appeal: {q['Main appeal']} [patient]",
        f"2. Image-derived anatomy: {anatomy}",
        f"3. Examination findings: {tagged['Oral Check']}",
        f"4. Diagnosis: {tagged['Diagnosis']}" + (f"\n   Categories: {', '.join(cats)}" if cats else ""),
        f"5. Treatment plan: {tagged['Treatment plan']}",
        "6. Similar cases used:\n   " + "\n   ".join(similar),
        "7. Items for clinician to verify:\n   - " + "\n   - ".join(verify),
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag_json", required=True)
    ap.add_argument("--mode", choices=["text", "facts", "facts_images"], default="facts_images")
    ap.add_argument("--base_model", default="google/medgemma-1.5-4b-it")
    ap.add_argument("--lora_adapter", default=None)
    ap.add_argument("--volume_dir", default=None)
    ap.add_argument("--mask_dir", default=None)
    ap.add_argument("--slice_index_json", default=None,
                    help="{case_id: [20 absolute axial indices]} recorded from the BiomedCLIP slice selection")
    ap.add_argument("--facts_dir", default=None, help="cache dir for facts text files")
    ap.add_argument("--min_component_voxels", type=int, default=100,
                    help="ignore mask components smaller than this many voxels")
    ap.add_argument("--label_map", default=None, help="JSON file, see DEFAULT_STRUCTURES comment")
    ap.add_argument("--slice_axis", type=int, default=2, help="axis of the 400-slice dimension in the NIfTI")
    ap.add_argument("--overview_size", type=int, default=896)
    ap.add_argument("--crop_half", type=int, default=80, help="half crop width in voxels")
    ap.add_argument("--crop_size", type=int, default=896)
    ap.add_argument("--save_images_dir", default=None, help="save overlays for clinician QC")
    ap.add_argument("--output_json", default="predictions.json")
    ap.add_argument("--output_reports", default="reports.json")
    ap.add_argument("--output_zip", default="submission.zip")
    ap.add_argument("--status_csv", default="generation_status.csv")
    ap.add_argument("--max_exemplars", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=700)
    ap.add_argument("--num_beams", type=int, default=1)
    ap.add_argument("--copy_threshold", type=float, default=0.5)
    ap.add_argument("--fallback", choices=["unable", "top_case"], default="unable")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.rag_json, "r", encoding="utf-8") as f:
        rag = json.load(f)
    if isinstance(rag, list):
        rag = {str(it.get("case_id", i)): it for i, it in enumerate(rag)}
    case_ids = list(rag.keys())[: args.limit] if args.limit else list(rag.keys())
    print(f"{len(case_ids)} cases, mode={args.mode}")

    label_map = json.load(open(args.label_map)) if args.label_map else {}
    slice_index = json.load(open(args.slice_index_json)) if args.slice_index_json else {}
    slice_index = {str(k): v for k, v in slice_index.items()}

    model, processor = load_model(args.base_model, args.lora_adapter)

    predictions, reports, rows = {}, {}, []
    counts = {"model": 0, "model_after_retry": 0, "fallback": 0}
    for n, cid in enumerate(case_ids, 1):
        case = rag[cid]
        try:
            inputs = prepare_case_inputs(cid, case, args, label_map, slice_index)
            tagged, clean, flags, status = generate_record(model, processor, case, inputs, args)
        except Exception:
            print(f"  [ERROR] case {cid}")
            traceback.print_exc()
            inputs = {"facts": None, "facts_text": None, "images": [], "notes": ["pipeline error"]}
            clean = tagged = {f: UNABLE for f in TARGET_FIELDS}
            flags, status = ["fallback: exception during generation"], "fallback"

        predictions[cid] = {f: clean[f] for f in TARGET_FIELDS}
        facts = inputs.get("facts")
        reports[cid] = {
            "status": status, "mode": args.mode,
            "fields_tagged": tagged, "flags": flags,
            "mismatch_teeth": facts["mismatch"] if facts else [],
            "n_images": len(inputs["images"]),
            "report": build_report(cid, case, tagged, flags, inputs, status, args),
        }
        rows.append({"Case ID": cid, "generation_status": status, "n_flags": len(flags),
                     "mismatch": bool(facts and facts["mismatch"]), "n_images": len(inputs["images"])})
        counts[status] += 1
        if n % 10 == 0 or n == len(case_ids):
            print(f"  {n}/{len(case_ids)}")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    with open(args.output_reports, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2, default=list)
    with open(args.status_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["Case ID"])
        w.writeheader()
        w.writerows(rows)
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(args.output_json, arcname="predictions.json")

    print(f"Wrote {args.output_json}, {args.output_reports}, {args.output_zip}")
    print(f"Status: {counts}")


if __name__ == "__main__":
    main()
