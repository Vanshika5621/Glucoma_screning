# Glaucoma Screening Research Dashboard

A local academic prototype for retinal fundus image classification, built
with React and FastAPI. It includes an upload dashboard, API, and modular
training/evaluation pipeline. No dataset or trained checkpoint is included;
predictions remain unavailable until you supply a suitable dataset, train a
model, and evaluate it.

## Important limitations

The eventual research task is binary **glaucoma** / **normal** classification
from retinal fundus images, with a model score. A regular selfie of the front
of the eye is not supported. The app is not a medical device and cannot
diagnose or rule out glaucoma; results must not guide care decisions. Contact
an eye-care professional about health concerns.

The current app validates image file encoding/dimensions. It does not recognize
fundus content or produce an actual prediction unless a compatible model
checkpoint is present.

When a compatible checkpoint is configured, predictions include a Grad-CAM
explanation: the original upload, a class-specific heatmap, and a blended
overlay. ResNet50 uses the final bottleneck convolution
`layer4[-1].conv3`; EfficientNet-B0 uses its last feature block. The
explanation is approximate and highlights
regions associated with the model's score; it does not prove glaucoma or show
clinically meaningful pathology. If Grad-CAM fails, the API still returns the
prediction and clearly reports that the visualization is unavailable. Returned
visuals are resized to a maximum 1024-pixel edge to limit response size.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for the architecture and build stages,
[PRD.md](./PRD.md) for product scope and acceptance criteria, and
[QA_TESTING_PLAN.md](./QA_TESTING_PLAN.md) for test cases, execution status,
Postman guidance, and privacy/release checks.

## Requirements

- macOS
- Python 3.11
- Node.js and npm
- Visual Studio Code (recommended)

Open the `glaucoma-screening` folder in VS Code. Commands below run from the
project folder unless the command first changes directory.

Check runtime versions:

```bash
python3.11 --version
node --version
npm --version
```

## Backend environment

Create and activate a Python virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch and torchvision are large and may take some time to install. Create a
local configuration file if you need to change defaults:

```bash
cp .env.example .env
```

`.env` is ignored by Git. Never put secrets in it.

## Frontend setup

```bash
cd frontend
npm install
```

## Run in development

Open two terminals in the project folder.

Terminal 1 — FastAPI:

```bash
source .venv/bin/activate
python -m uvicorn app.main:app --app-dir backend --reload
```

Terminal 2 — React:

```bash
cd frontend
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://127.0.0.1:5173`).
For a production-style local build, run `npm run build` in `frontend/`; then
start FastAPI and open `http://127.0.0.1:8000`. FastAPI serves the React build
from `frontend/dist`. API documentation is at `http://127.0.0.1:8000/docs`.

## API

- `GET /api/health` reports application and model availability separately.
- `GET /api/metrics` serves a real local evaluation report or explains that
  metrics are not available.
- `POST /api/images/validate` checks actual JPEG/PNG encoding and dimensions.
  It does not verify that the image is retinal and does not persist the upload.
- `POST /api/predict` returns a research-model score only when a compatible
  checkpoint is configured; otherwise it returns HTTP 503. Successful
  responses include the original image, Grad-CAM heatmap, and overlay as PNG
  data URLs, or an explicit visualization-unavailable message if explanation
  generation fails.

The local server may temporarily buffer request data while processing it. Do
not expose the development server to the public internet.

## Dataset and training

Select and inspect a public fundus dataset manually. Check its current license,
terms, intended use, citation, label meaning, image population, and whether
patient identifiers are available. Do not automatically download data or
commit datasets to this repository; local `data/` is Git-ignored.

Prepare a CSV manifest, with image paths relative to that CSV:

```csv
image_path,label,patient_id
images/example_001.png,normal,subject_001
images/example_002.png,glaucoma,subject_002
```

Labels must be `normal` or `glaucoma` (case-insensitive). Include a complete
`patient_id` column to request deterministic stratified patient-wise splits;
all images for one patient stay in one partition. Missing or blank IDs are
rejected rather than silently treated as independent images. If the column is
absent, the pipeline uses stratified image-level splits and records that
limitation. Small datasets may not support reliable three-way splitting; the
pipeline rejects splits that omit a class or cannot keep both classes in each
patient-wise partition.

The shared preprocessing pipeline decodes EXIF orientation, converts to RGB,
preserves aspect ratio with centered padding to 224 × 224, then applies
ImageNet normalization. Training-only augmentation is kept separate from
validation, test, and inference transforms. Checkpoints record the
preprocessing version and incompatible checkpoints are not served.

Train the default ResNet50 model:

```bash
.venv/bin/python -m src.glaucoma_ml.training.train \
  --manifest data/processed/REFUGE/train_only_manifest.csv \
  --dataset-name "REFUGE-training-partition" \
  --architecture resnet50 \
  --epochs 30 \
  --batch-size 8 \
  --seed 42
```

By default the model initializes from torchvision's default ImageNet weights;
the first run may download them. It fine-tunes the backbone and classifier
using AdamW, balanced training-split class weights, ReduceLROnPlateau on
validation loss, and early stopping. Use `--no-pretrained` to train from
scratch. The default split is stratified 70/15/15 by image (or by patient if a
complete `patient_id` column is supplied). A manifest containing all three
REFUGE official partitions preserves those partitions; a train-only manifest
is split internally. For class counts, epoch plots, and metrics to be
meaningful, start from a manifest limited to the intended training source.

Each run gets a new directory under `runs/` (or use `--output-dir`). It saves
`best_model.pt`, `training_config.json`, `training_history.csv`, `metrics.json`,
`split_manifest.csv`, separate train/validation accuracy and loss plots, plus
confusion matrix and ROC curve. A non-empty run directory is never overwritten.
Training metrics are computed after each epoch on the unaugmented training
images. The training command also evaluates its internal/official test
partition once and saves that result in the run directory.

To compare architectures, run each with the same manifest, seed, split
fractions, epochs/patience policy, and batch size, changing only
`--architecture resnet50` to `--architecture efficientnet_b0`. Select training
decisions using validation metrics; do not choose a model or tune its threshold
from test metrics. A patient-wise split is only possible if genuine patient
IDs are available; the inspected REFUGE metadata does not establish such IDs.
Image-level splitting must be reported as a limitation.

Metrics are computed from the held-out test split. Sensitivity, specificity,
precision, recall, F1, accuracy, ROC-AUC, confusion matrix, ROC curve, and
training/validation curves are included where computable. These are research
metrics, not proof of clinical performance; review dataset provenance,
split quality, and generalization before displaying results.

### Evaluate the existing checkpoint on REFUGE validation

This command performs inference/evaluation only: it does not train, update
weights, or tune a threshold. By default, it reads only `official_validation`
fundus photos/labels from the generated inspection metadata and writes a
separate report directory. It refuses to overwrite an existing report.

```bash
python -m src.glaucoma_ml.evaluation.evaluate_checkpoint \
  --metadata data/processed/REFUGE/metadata.csv \
  --checkpoint models/best_model.pt \
  --output-dir reports/refuge-official-validation \
  --batch-size 8
```

The official validation set is a separate dataset partition; do not use it to
select epochs or thresholds if reporting it as validation evaluation. This
research result is not external clinical validation.

After all model and threshold decisions are complete, the official test
partition can be evaluated once with the same command by changing the split:

```bash
python -m src.glaucoma_ml.evaluation.evaluate_checkpoint \
  --metadata data/processed/REFUGE/metadata.csv \
  --checkpoint models/best_model.pt \
  --split official_test \
  --batch-size 8
```

This writes to `reports/refuge-official-test/` by default. Do not use the test
result to tune the model; it is the final held-out evaluation, not clinical
validation.

## CPU and Apple Silicon

Training/inference uses Apple Silicon MPS when PyTorch reports it available;
otherwise it falls back to CPU. MPS depends on the Mac, macOS version, and
PyTorch build. CPU works but may be slow for ResNet50. No CUDA GPU is required.

## Tests

Backend tests:

```bash
python -m pytest
```

Frontend tests and type/build checks:

```bash
cd frontend
npm test
npm run build
```

The test suite checks API validation, unavailable-model and unavailable-metrics
states, and patient-wise split isolation. Tests do not establish clinical
validity.

## Project structure

- `backend/app/` — FastAPI routes, schemas, configuration, and services
- `frontend/` — React, Tailwind CSS, Axios, and Recharts dashboard
- `src/glaucoma_ml/` — image data, models, training, and evaluation
- `data/raw/`, `data/processed/` — user-provided data, ignored by Git
- `models/` — local checkpoints, ignored by Git
- `reports/` — local evaluation output
- `tests/` — backend tests

This single-user dashboard has no login or database because it does not need
accounts, history, or saved predictions. Images are not retained by the
application after a request.
