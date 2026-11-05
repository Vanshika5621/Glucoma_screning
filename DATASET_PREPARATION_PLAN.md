# Dataset Preparation and Preprocessing Plan

## Status and purpose

**Status:** REFUGE was selected for a local experiment and inspected read-only.
Training and official validation/test evaluation have since been run; actual
inspection findings are recorded in
[`data/processed/REFUGE/inspection_report.md`](./data/processed/REFUGE/inspection_report.md),
and measured evaluation results are recorded in
[`QA_TESTING_PLAN.md`](./QA_TESTING_PLAN.md). This plan remains the workflow
and decision record; it is not evidence of dataset reuse permission.

The local copy came from the Kaggle listing
[`sanjidkhan/refuge-glaucoma-dataset`](https://www.kaggle.com/datasets/sanjidkhan/refuge-glaucoma-dataset),
which declares CC BY-SA 4.0. The uploader's authority to relicense upstream
REFUGE content has not been verified. Supervisor/rights-holder confirmation
remains outstanding; do not redistribute source images or weights. The local
inspection found 1,200 fundus photographs across three official partitions
(400 per partition; each contained 360 normal and 40 glaucoma images), decoded
3,200 total image assets without corruption, and found no documented patient
IDs. See the inspection report for methods and caveats.

This plan describes the evidence and implementation work required before
training the academic glaucoma fundus-image classifier. It follows
[PRD.md](./PRD.md): local research use, no login/database, no fabricated
results, and no diagnostic claims. It is not a clinical protocol.

**Review date:** 2026-11-05. Dataset availability and reuse terms can change.
The project team must inspect the exact current source and package terms before
using any data. A paper, challenge page, Kaggle listing, or public download
does not by itself establish permission to use, modify, or redistribute the
images.

## 1. Dataset selection

### Comparison

| Candidate | Task and labels to verify | Availability and licensing | Fit and risks |
|---|---|---|---|
| **REFUGE** | A glaucoma fundus challenge dataset. The challenge describes glaucoma assessment and related optic-disc/cup and fovea tasks. Confirm which released partition actually has usable binary labels, the label/reference-standard definitions, and whether labels are per image, eye, or subject. | The challenge has an official project page and a dataset access/download workflow. Access to the download page may require registration or be restricted. No blanket open-data or redistribution license should be inferred; record the exact terms shown for the downloaded release. Patient IDs and any permitted reuse must be confirmed from the package documentation. | **Recommended first candidate, subject to the gate below.** It is the closest match to this project's fundus/glaucoma task and has a challenge context. A challenge score or internal test result is not external or clinical validation. |
| **ORIGA / ORIGA-light** | Published for glaucoma analysis from color fundus images and includes glaucoma-related reference labels; confirm the exact release, label definition, and whether the release provides only binary diagnoses or additional optic-disc/cup annotations. Do not equate any segmentation annotation with a confirmed diagnosis. | Locate the original research source or contact the rights holder. Availability may be through research access or a specific distribution rather than an unrestricted download. No reuse, modification, commercial-use, or redistribution permission is assumed without the actual release terms. Verify patient identifiers and their permitted use. | Potential secondary research dataset or external evaluation set if access and compatible labels/permissions are confirmed. Different image sources and populations from training can make it useful for a separate generalization check, but dataset shift must be reported. |
| **RIM-ONE family, including RIM-ONE DL** | Several releases exist. Select and name one exact release. Verify whether its classes are healthy/glaucoma, include suspect/other categories, and how labels were assigned. Confirm whether labels are image-, eye-, or patient-level. | Use the release-specific research/author source and its terms. Do not assume that one version's access conditions apply to another or to third-party mirrors. Record the exact package/version, license or written permission, and restrictions. Confirm patient-ID availability. | Could be a manageable academic candidate if a suitable release and terms are confirmed. Version differences, differing label conventions, and small or nonrepresentative samples can limit conclusions and make it unsuitable to merge blindly with another dataset. |

### Recommendation and selection gate

Start by evaluating **REFUGE** as the primary candidate, not by assuming it is
already selected or approved. Before adopting it, the team must verify:

1. The exact downloadable release and its current academic-use, storage,
   derivative-work, publication, and redistribution terms.
2. Which examples have released labels suitable for the project's two classes,
   and what clinical/reference process those labels represent.
3. Whether patient or subject IDs are supplied and may be used for splitting.
4. Whether enough labeled subjects and both classes are present to create
   train, validation, and test partitions without leakage.
5. Whether the image type, resolution, population, and acquisition setup fit
   the intended fundus-only research task.

If any gate fails, do not train on that release. Compare ORIGA-light and a
specific RIM-ONE version using the same checklist. Prefer one well-documented
dataset for the initial experiment over concatenating incompatible datasets.
If combining datasets is later justified, preserve dataset identity in every
manifest row and report dataset-specific performance; do not let images from
the same person cross partitions.

If no candidate's terms, usable labels, or patient grouping are sufficiently
clear, pause training and ask the dataset maintainer or project supervisor.
Do not substitute an unverified Kaggle mirror.

### Dataset source references

- [REFUGE challenge page](https://refuge.grand-challenge.org/)
- REFUGE challenge paper: Orlando et al., *REFUGE Challenge: A unified
  framework for evaluating automated methods for glaucoma assessment from
  fundus photographs*, Medical Image Analysis (2019),
  [doi:10.1016/j.media.2019.101570](https://doi.org/10.1016/j.media.2019.101570)
- ORIGA-light paper: *ORIGA-light: An online retinal fundus image database for
  glaucoma analysis and research*,
  [doi:10.1109/IEMBS.2010.5626137](https://doi.org/10.1109/IEMBS.2010.5626137)
- RIM-ONE: consult the exact release's official research/author distribution
  and cite that release's documentation in the dataset card. Third-party
  mirrors are discovery leads, not proof of provenance or reuse rights.

These references identify research sources; they do not grant data-use
permission.

## 2. Dataset organization

Keep the original dataset archive and files immutable. Do not place data in
Git, the web application, or a public folder. Store a local dataset-specific
working copy under the already ignored `data/` directory:

```text
data/
├── raw/
│   └── <dataset_id>/
│       ├── original/                  # exact source files; read-only after intake
│       └── SOURCE_NOTES.txt           # source URL, access date, terms location
├── processed/
│   └── <dataset_id>/
│       ├── images/
│       │   ├── normal/
│       │   └── glaucoma/
│       ├── splits/
│       │   ├── train.csv
│       │   ├── validation.csv
│       │   └── test.csv
│       ├── views/                     # optional derived train/val/test class folders
│       │   ├── train/{normal,glaucoma}/
│       │   ├── validation/{normal,glaucoma}/
│       │   └── test/{normal,glaucoma}/
│       ├── audit/
│       │   ├── image_audit.csv
│       │   ├── duplicate_review.csv
│       │   └── class_distribution.csv
│       └── visualizations/
└── metadata/
    └── <dataset_id>/
        ├── dataset_card.yaml
        ├── source_manifest.csv
        ├── preprocessing.yaml
        └── split_manifest.csv
```

The **CSV manifests are authoritative** for image labels, patient grouping,
and partition assignment. The class directories and optional split folder
views are derived conveniences, never independent label sources. They may
contain copies or links only if each row can be traced to the immutable source
and split checks pass. Never modify the source images in place.

### Required metadata

`dataset_card.yaml` should record, at minimum:

- dataset name, exact release/version, source/official URL, citation, access
  date, and local archive checksum;
- license/terms URL or permission record, allowed purpose, restrictions,
  attribution obligations, and redistribution status;
- label definitions, reference standard, source label fields, binary mapping,
  excluded/ambiguous classes, and known limitations;
- population/cohort description and acquisition/site/device information when
  supplied by the source;
- whether patient/subject identifiers exist, whether use for splitting is
  permitted, and how locally stored IDs are pseudonymized;
- intake date, preprocessing version, split seed/algorithm, split ratios or
  group counts, and responsible project members.

Each manifest row should retain provenance and make split audit possible:

```text
image_path,label,patient_id,eye,source_split,source_label,image_sha256,split
```

Only include `patient_id`, `eye`, or `source_split` when provided and permitted.
Use a project-local pseudonymous patient key if allowed; do not include direct
identifiers. The minimum model loader contract remains `image_path,label`.
Unknown values must be explicit, not guessed from filenames.

## 3. Intake and data validation workflow

Run the following checks before splitting or training. Write an audit report
and preserve rejected rows with reason codes; never silently drop data.

1. **Inventory and terms:** Record archive/source/version, file listing,
   source-provided split, label files, metadata files, and terms. Calculate a
   checksum for the downloaded archive. Confirm that local storage and
   processing are allowed.
2. **Manifest reconciliation:** Match each image to exactly one label using
   the source's documented key. Report image files without labels, labels
   without files, repeated keys, blank paths, unknown classes, and rows with
   incomplete patient IDs. Do not infer labels from folder names alone without
   verifying the dataset documentation.
3. **Decode and format:** Check actual file bytes rather than extension.
   Record detected format, dimensions, color mode, orientation metadata, and
   decode status. Confirm supported formats with the selected pipeline. Fully
   load each image so truncated/corrupt files are detected, not just headers.
4. **Image properties:** Flag zero/tiny dimensions, unsupported bit depth,
   unexpected animation/multiple frames, alpha/transparency, and suspicious
   grayscale inputs for review. Define a consistent handling policy before
   model training. Preserve valid color fundus images; do not silently convert
   all unusual cases.
5. **Corrupt images:** Exclude files that cannot be decoded safely from all
   splits, record path/hash and a reason, and report exclusion counts by class
   and source group. Do not repair files unless the repair is reproducible and
   permitted; if repaired, keep the original and record the transformation.
6. **Exact duplicates:** Compute SHA-256 over original file bytes. Flag exact
   duplicates, including across source partitions and labels. Resolve
   conflicting labels before splitting. A duplicate image must never appear in
   two splits.
7. **Near duplicates:** Optionally compute perceptual hashes or image
   embeddings to flag re-encodings, crops, or repeat captures. Treat these as
   review candidates, not automatic deletion. Group confirmed repeat images
   by patient/session before splitting.
8. **Label review:** Check every source label maps to `normal` or `glaucoma`
   under a documented rule. Exclude or separately retain `suspect`,
   ungradable, uncertain, or contradictory labels; do not force them into
   either binary class. Preserve original label and mapping in the audit.
9. **Class and cohort profile:** Report image and unique-patient counts by
   class, source partition, eye side, site/device/population when known, and
   exclusion reason. Make no claims about balance or sample adequacy until
   measured from the selected release.
10. **Human review:** Review a stratified sample of original images and
    labels, especially exclusions, duplicates, and unexpected properties.
    Record reviewer, date, and outcome without exposing personal identifiers.

## 4. Data splitting and leakage prevention

### Recommended initial split

Use a **70% training / 15% validation / 15% test target** only if the
available data support all three partitions with both classes and enough
independent patients. These percentages are planning defaults, not a promise
that a small dataset can support them. If a candidate dataset has an official
split, preserve it only after confirming its construction, access terms, and
label availability; do not merge official validation/test images into training
without a documented reason.

### Patient-wise rules

- Keep all images from one patient/subject—including both eyes, repeat visits,
  and duplicate or near-duplicate captures—in one partition.
- Determine the correct grouping unit from dataset documentation. Do not
  assume one filename equals one patient or that an eye identifier is a
  patient identifier.
- Prefer a deterministic **stratified group** assignment, balancing labels at
  the group level as feasible. Validate class representation and patient
  disjointness after assignment.
- If an individual has multiple image labels, define how that affects group
  stratification without overwriting the image-level targets. Report
  conflicting subject-level labels.
- Lock the test partition before model selection. Use training and validation
  only for model fitting, early stopping, and any threshold selection. Evaluate
  the test set once for the selected experiment.
- Keep any separately sourced dataset as a distinct external evaluation
  cohort if its terms permit. Do not tune on it and do not call it clinical
  validation.

If usable patient IDs are missing, use a seeded stratified image-level split
only when permitted and feasible. Mark the experiment **patient leakage
unknown**; disclose that multiple images from one person may cross splits.
Do not claim patient-wise validation.

### Current implementation baseline and remaining dataset validation

The current manifest loader accepts `image_path`, `label`, and optional
`patient_id`. Patient-level splitting now searches deterministic
`StratifiedGroupKFold` candidates for partitions that retain both classes and
approach the requested sample/group fractions. It rejects missing/blank IDs
and raises an actionable error if a patient-wise split cannot retain both
classes. The selected dataset may still be too small or imbalanced for this
strategy; verify feasibility with its real labels before training. Never fall
back silently to an image-level split.

Treat `train.csv`, `validation.csv`, and `test.csv` as frozen, reviewable
artifacts after split creation. Record seed, algorithm/version, class counts,
patient counts, and input-manifest hash. Update the training loader to consume
these persisted partitions rather than independently recreating splits on
every run. This persistence/audit work remains to be implemented.

## 5. Image preprocessing

Define one versioned preprocessing specification shared by training,
validation, test, visualization, and inference.

1. **Decode/orientation:** Decode with Pillow, apply EXIF orientation once,
   fully load, and convert to RGB. Handle palette and grayscale sources
   explicitly. Reject or consistently composite alpha channels according to
   the selected dataset policy. Do not use OpenCV BGR ordering as model input.
2. **Color:** Keep three-channel RGB. Do not drop channels or convert to
   grayscale by default; color/illumination information may carry useful
   signal. Record any conversion.
3. **Resize:** Use 224 x 224 as the initial ResNet50/EfficientNet-B0 baseline
   size unless the selected dataset/model experiment justifies another size.
   Do not crop away the optic nerve or field of view. Prefer an
   aspect-ratio-preserving resize with a documented, consistent pad policy
   over silently stretching the retinal field. Inspect padding and field-of-
   view effects on samples before freezing the choice. If direct square
   stretching is retained for comparison, document and test it as a distinct
   experiment.
4. **Intensity:** Convert to floating point in `[0, 1]`, then normalize RGB
   channels with the ImageNet mean `(0.485, 0.456, 0.406)` and standard
   deviation `(0.229, 0.224, 0.225)` for the pretrained torchvision baseline.
   Apply these same constants at validation, test, and inference. If
   calculating dataset-specific statistics for a separate experiment, use
   training images only and persist the values.
5. **Enhancement:** Do not apply CLAHE, histogram equalization, green-channel
   extraction, sharpening, denoising, or gamma correction in the baseline.
   Such operations can amplify artifacts or change clinical appearance. Add
   one only as a separately versioned ablation with a written rationale,
   train-only fitting where applicable, paired visual review, and identical
   train/validation/test/inference use.
6. **Order:** Decode/orient -> RGB policy -> geometry -> tensor conversion ->
   normalization. Store the exact order, parameters, interpolation, pad fill,
   and library versions in `preprocessing.yaml` and checkpoint metadata.

### Current implementation baseline and visual-review gate

The current implementation uses one shared preprocessing module for training
and API inference. It applies EXIF orientation, RGB conversion,
aspect-ratio-preserving resize with centered black padding to 224 x 224, then
tensor conversion and ImageNet normalization. It no longer resizes the same
image twice. This is a provisional baseline only: inspect paired samples from
the selected dataset for field-of-view and padding effects before training,
and version any resulting policy change in checkpoint metadata. Augmentation
is defined separately and enabled only for the training dataset.

## 6. Training-only data augmentation

Augmentation is applied dynamically to **training images only**. Never
augment validation or test data, and never write augmented files back over
the source images.

Start conservatively:

- **Rotation:** Random small in-plane rotation (initial range approximately
  `-10` to `+10` degrees) with a fixed, documented fill policy. Confirm that
  orientation is not a label cue for the selected dataset.
- **Horizontal flip:** Optional 50% probability only after confirming the
  task label is invariant to left/right eye laterality and no downstream
  laterality output is required. Do not infer laterality after flipping.
- **Brightness/contrast:** Small jitter (initially around 8-10%) to reflect
  acquisition variation without washing out disc/cup detail. Review side by
  side for loss of clinically relevant structures.
- **Saturation:** At most subtle jitter (for example 5%) if visual review
  shows color remains plausible. Avoid hue shifts by default.
- **Avoid by default:** Vertical flips, large rotations that alter expected
  orientation without justification, strong crops/zoom, elastic warps,
  perspective transforms, blur, heavy noise, and synthetic color changes.
  These may remove or invent cues.

All augmentation values are experiment configuration, not assumed universal
medical standards. Save the random seed and configuration. If classes are
imbalanced, first report the imbalance; consider training-only class weights
or a sampler before stronger augmentation. Never rebalance validation/test by
duplicating samples.

The current training transforms already use horizontal flip, an 8-degree
rotation, and modest brightness/contrast/saturation jitter. Retain them only
after the dataset-specific invariance and visual-review checks above. Keep
validation/test transforms deterministic.

## 7. Visualization and review artifacts

Generate local reports from the validated manifest and frozen splits:

- class counts for images and unique patients, shown overall and per split;
- count of missing/ambiguous labels, corrupt files, exact duplicates, and
  reviewed near-duplicate groups;
- a fixed-seed montage of original images by class and split, with no direct
  identifiers in captions;
- paired original and preprocessed images showing the full field of view,
  orientation, padding/cropping, and RGB appearance;
- a small augmentation montage showing each transformation beside its
  original;
- optional image-size, aspect-ratio, and color summaries to reveal acquisition
  differences.

Visualizations are quality-control aids, not evidence that images are
diagnostic or that labels are correct. Do not show patient identifiers, store
images in frontend assets, or commit generated patient imagery. Keep artifacts
inside ignored local `data/processed/<dataset_id>/audit/` or
`visualizations/`.

## 8. Reproducibility and documentation

For each experiment, preserve:

- dataset card and a human-reviewed terms/license record;
- original archive checksum and source/version;
- manifest and mapping-rule version;
- image audit and exclusion reasons;
- exact/group duplicate review results;
- frozen split manifests and hash, seed, group strategy, and patient overlap
  check;
- preprocessing and augmentation configuration;
- Python/PyTorch/torchvision/Pillow/OpenCV/scikit-learn versions, device
  (CPU/MPS), and training arguments;
- checkpoint metadata and evaluation report that refer to the same dataset,
  split manifest, class order, and preprocessing version.

Do not commit data, personal identifiers, licensed images, checkpoints,
environment secrets, or generated sample images. Commit code and non-sensitive
schemas/documentation only. Cite the chosen dataset in any report or
presentation according to its terms.

## 9. Implementation plan (before code changes)

No code changes or data downloads are part of this plan. Implement in this
order after the team chooses and approves a dataset:

### Phase 0 — Dataset decision and permission

- Compare exact releases using Section 1.
- Review current source terms and get supervisor/maintainer clarification for
  ambiguous rights.
- Record the selected release, citation, label/reference definitions,
  patient-ID availability, and allowed processing in a completed dataset card.
- **Gate:** no download or training until the team can document permission and
  label suitability.

### Phase 1 — Intake, manifest, and audit

- Add a dataset-specific intake manifest mapping source IDs to image paths,
  source labels, optional permitted patient grouping, and provenance.
- Implement deterministic image decode/format/dimension checks and audit
  output.
- Add SHA-256 exact duplicate detection and a review-only near-duplicate
  workflow; resolve conflicting labels.
- Report exclusions and class/patient distribution without embedding data in
  the repository.
- **Gate:** every included image has one traceable label; exclusions and
  duplicates are reviewed.

### Phase 2 — Leakage-safe frozen splits

- Implement deterministic stratified patient/group splitting and persisted
  train/validation/test CSVs; add overlap and repeat-image checks.
- Add tests for repeated patients, both-eye cases, mixed patient labels,
  missing IDs, duplicate images, rare classes, and too-small cohorts.
- For absent identifiers, require an explicit image-level split setting and
  record leakage uncertainty in all reports.
- **Gate:** split counts/classes are feasible, patient sets are disjoint when
  possible, and test rows are locked.

### Phase 3 — Unified preprocessing and visualization

- Consolidate one shared decoder/transform path for offline and API inference.
- Resolve aspect ratio/padding and transparency policy with paired sample
  visualizations before training.
- Version and serialize size, channel, interpolation, normalization, and
  augmentation settings into the checkpoint/report metadata.
- Test that validation/test transforms are deterministic and training-only
  augmentations do not alter labels.
- **Gate:** visual review is signed off and training/inference transforms
  match.

### Phase 4 — Baseline training and report

- Train only after Phases 0-3 pass; start with ResNet50 and documented
  torchvision initialization settings.
- Select checkpoints/epochs using validation data only.
- Evaluate once on the frozen test set; include class-wise counts and metrics,
  split provenance, limitations, and dataset citation.
- Keep EfficientNet-B0 as an optional controlled comparison, not a requirement
  to ship two models.
- **Gate:** report is derived from real held-out outputs and does not claim
  clinical or external validation.

### Phase 5 — Review and later integration

- Review dataset terms, audit outputs, split checks, preprocessing,
  evaluation, and medical wording with the project supervisor.
- Only then consider enabling a compatible local inference checkpoint in the
  existing dashboard.
- Do not add a fabricated sample prediction or demo metrics if any gate is
  incomplete.

## 10. Acceptance criteria for dataset readiness

- Exact dataset release and terms are documented; academic processing is
  permitted.
- Every included image is reconciled to an unambiguous source label and
  provenance row.
- Corrupt files, missing labels, unsupported formats, exact duplicates, and
  near-duplicate review outcomes are accounted for.
- Class and patient distributions are computed from the selected data, not
  copied from assumptions or a third-party mirror.
- The frozen split manifests are deterministic; patients and exact duplicate
  images do not cross partitions.
- If patient IDs are unavailable, image-level splitting is explicitly labeled
  as a limitation and no patient-wise claim is made.
- Preprocessing preserves a documented color/channel and field-of-view policy
  and is identical across training, evaluation, and inference.
- Augmentation is confined to the training loader and passes visual review.
- Re-running intake/splitting with the same inputs and versions reproduces
  the audit and split assignments, or records the source of any expected
  nondeterminism.
- No downloaded data, trained weights, invented counts, or unverified
  performance claims are committed.
