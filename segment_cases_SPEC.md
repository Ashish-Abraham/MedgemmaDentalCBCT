# Task: write `segment_cases.py` — batch CBCT tooth-segmentation inference

## Goal
Run a trained U-Mamba2 (nnU-Net v2.6.2 fork) model over a folder of CBCT volumes and
produce per-case tooth masks that a downstream script (`generate_predictions_with_seg.py`)
consumes. Single Python file, stdlib + numpy/nibabel/scipy (+ matplotlib only for optional
QC PNGs). The heavy inference is delegated to the installed `nnUNetv2_predict` CLI via
subprocess — you do NOT reimplement preprocessing, sliding-window, or resampling.

## Downstream contract (must hold exactly)
The downstream script does, per case:
  - looks up `OUT_DIR/<case_id>.nii.gz` (or `.nii`); flat directory, no subfolders
  - requires mask NIfTI shape == volume NIfTI shape (it raises otherwise)
  - reads labels as integers, 0 = background
  - tooth label value must equal the FDI number (11–48) by default; if the model's raw
    values are not FDI, a label_map JSON `{"teeth": {"<FDI>": <mask_value>}, "structures": {...}}`
    is passed downstream as `--label_map`
  - physical spacing must come from the NIfTI header (used for mm distances)
`nnUNetv2_predict` already writes predictions in the input's original geometry
(shape, affine, spacing), so the contract holds as long as you do not resample afterwards.

## Environment facts
- The U-Mamba2 repo (https://github.com/zhiqin1998/U-Mamba2) is installed with
  `pip install -e .` in the active Python env; its `nnUNetv2_predict` entry point exists
  (check `shutil.which("nnUNetv2_predict")`, fall back to
  `python -m nnunetv2.inference.predict_from_raw_data`).
- Trained model lives under a nnU-Net results root, layout:
  `<results_root>/Dataset<XXX>_<Name>/<Trainer>__<Plans>__<Config>/fold_<k>/checkpoint_best.pth`
  plus `plans.json` and `dataset.json` next to the fold dirs.
- Run `nnUNetv2_predict --help` at build time to confirm exact CLI flag names for the
  installed version (expect: -i -o -d -c -f -tr -p -device --disable_tta --overwrite;
  parallelism may be -npp/-nps or -nproc).

## CLI
```
python segment_cases.py
  --volume_dir DIR              # <case_id>.nii.gz per case (case_id == RAG json keys)
  --results_root DIR            # nnU-Net results root
  --model_dir DIR               # optional: point directly at the <Trainer>__<Plans>__<Config> folder
  --dataset / --config / --trainer / --plans   # overrides if auto-detection is ambiguous
  --out_dir DIR                 # default: mask_dir
  --label_map_out FILE          # default: <out_dir>/../label_map.json; only written if needed (below)
  --cases FILE                  # optional: one case_id per line (subset); default = all volumes
  --folds all                   # or e.g. "0"; default all
  --nproc 2 --device cuda --disable_tta          # passed through to nnUNetv2_predict
  --min_component_voxels 100     # per-label speckle filter (postprocess)
  --keep_largest                # optional: per label keep only largest component (default off)
  --work_dir DIR                # staging + raw predictions (default: seg_work/)
  --qc_dir DIR                  # optional QC overlay PNGs
  --manifest FILE               # default: <out_dir>/segmentation_manifest.csv
  --overwrite --clean_work --copy   # resume/behavior flags
```

## Step 0 — resolve the model
1. If `--model_dir` given, use it; else glob `<results_root>/Dataset*/*__*__*` containing
   `fold_*` dirs. Exactly one match -> use it. Multiple -> fail with the list and require
   explicit `--dataset/--config/--trainer/--plans`.
2. Parse trainer/plans/config from the folder name (split on `__`), dataset id from the
   `Dataset<XXX>_<Name>` parent.
3. Set `os.environ["nnUNet_results"] = results_root` for the subprocess.
4. Read `<model_dir>/dataset.json` -> `labels` dict (name -> int value). Build:
   - `is_fdi(name)`: `re.fullmatch(r"[1-4][1-8]", name)`
   - teeth: names that are FDI. If value == int(name) for all -> IDENTITY mapping
     (no teeth entry needed downstream). Otherwise record fdi -> raw_value for label_map.
   - structures: remaining names, lowercased; name containing "canal" + "right"/"r" ->
     canal_right, "canal"+"left" -> canal_left, "sinus"+"right"/"left" similarly.
     Unmatched names: print warning, ignore.
   - If label values are lists (nnU-Net "regions" mode) -> abort with a clear message
     (not expected for this model).

## Step 1 — stage inputs
1. Enumerate cases: `--cases` file if given, else all `*.nii.gz` (and `.nii`) in
   `--volume_dir`, case_id = filename minus extension. Sanity-check: no spaces, no `__` in
   case_id; warn on dots (nnU-Net strips trailing `.0000` carefully).
2. Skip a case if `<out_dir>/<case_id>.nii.gz` exists and not `--overwrite` (resume).
3. Create `<work_dir>/staging/`; for each remaining case, symlink
   `<volume_dir>/<case_id>.nii.gz` -> `<work_dir>/staging/<case_id>_0000.nii.gz`
   (the `_0000` channel suffix is mandatory for nnU-Net; copy instead of symlink with
   `--copy`). Also create `<work_dir>/raw_predictions/`.

## Step 2 — run prediction
One subprocess call for the whole batch:
```
nnUNetv2_predict -i <work_dir>/staging -o <work_dir>/raw_predictions
  -d <dataset_id> -c <config> -tr <trainer> -p <plans> -f <folds>
  [-device cuda] [--disable_tta] [--overwrite] [parallelism flags]
```
with `nnUNet_results` set in env. Stream stdout/stderr to console and tee to
`<work_dir>/predict.log`.
If the batch call fails: fall back to per-case invocation (staging dir with one file) to
isolate failures; record per-case status; never abort the whole run.

## Step 3 — postprocess per successful case
1. Load prediction `<work_dir>/raw_predictions/<case_id>.nii.gz` with nibabel.
2. Load the original volume; assert `pred.shape == vol.shape` -> if not, mark case FAILED
   with reason, continue.
3. If any tooth mapping is non-identity, remap via a lookup table
   (`lut = np.zeros(max_label+1, dtype=uint16); lut[raw] = fdi`).
4. Speckle filter: for each unique nonzero label, `scipy.ndimage.label` on
   `(mask == label)`; drop connected components with < `--min_component_voxels` voxels
   (zero them out). With `--keep_largest`, keep only the largest component per label.
5. Cast to uint8 if max value <= 255 else uint16. Save
   `<out_dir>/<case_id>.nii.gz` reusing the prediction's nibabel image (correct affine,
   header, zooms) — `nib.Nifti1Image(data, pred_img.affine, pred_img.header)`, set
   `set_data_dtype`.
6. Manifest row: case_id, status(ok/failed/skipped), n_labels_found,
   teeth_upper, teeth_lower (count FDI 1x/2x vs 3x/4x), structure labels found,
   n_speckle_components_dropped, remapped(yes/no).

## Step 4 — label_map.json
Write `--label_map_out` ONLY if (a) any tooth raw value != FDI, or (b) structure labels
were matched. Format exactly:
```json
{"teeth": {"36": 26, "...": 0}, "structures": {"canal_right": 101, "canal_left": 102}}
```
- `teeth` maps FDI string -> value present in the SAVED masks (omit the section entirely
  for identity mapping).
- `structures` maps canonical names -> raw values.
If neither: do not write the file and print that downstream needs no `--label_map`.

## Step 5 — QC overlays (only if `--qc_dir`)
Per case: pick the middle axial slice (axis with the largest dim), render grayscale
volume slice + colored contour per label (matplotlib), save
`<qc_dir>/<case_id>.png`.

## Step 6 — final verification pass + summary
For every case: reload volume + final mask, assert shape equality; assert every nonzero
mask value is in the expected label set (FDI + structure raw values); update manifest
row with `verified=yes/no`. Print summary table (n ok / failed / skipped) and, finally,
print the ready-to-run downstream command with all paths filled in, e.g.:
```
python generate_predictions_with_seg.py --mask_dir <out_dir> \
  [--label_map <label_map_out>] --volume_dir ... --rag_json ...
```
Exit code 1 if any case failed verification.

## Hard constraints
- Never resample/crop/pad masks after nnU-Net writes them (geometry contract).
- Never save probabilities or float arrays.
- One case failing must not stop the batch; everything resumable by re-running
  (existing outputs skipped without `--overwrite`).
- Do not modify the U-Mamba2 repo, `generate_predictions_with_seg.py`, or retrain anything.
- No comments/docstrings needed; concise typed functions.

## Function layout (suggested)
`find_cases`, `resolve_model` (returns dataset_id, config, trainer, plans, labels dict),
`build_label_mapping(labels_dict)` (returns remap_lut, label_map_out_dict, canonical set),
`stage_inputs`, `run_predict(batch) -> per-case status`, `postprocess_case(case_id)`,
`qc_overlay(case_id)`, `verify_case(case_id)`, `main`.

## Acceptance checks (run these before declaring done)
1. On 2 real volumes: saved masks have shape == input, integer dtype,
   `np.allclose(mask_img.affine, vol_img.affine)`.
2. Synthetic speckle-filter unit test: a 125-voxel blob of label 36 survives; 4-voxel
   blob of 36 vanishes; a speckle-only label yields no voxels.
3. Re-run -> all cases report skipped (idempotent resume).
4. label_map.json (if produced) verified against dataset.json by hand.
5. Downstream smoke test: `python generate_predictions_with_seg.py --mode facts
   --rag_json <rag> --mask_dir <out_dir> --volume_dir <volumes> --limit 1`
   runs without shape/dtype errors and prints teeth detected.
