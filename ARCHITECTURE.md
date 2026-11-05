# System Architecture

## 1. Scope and architecture decisions

This document describes the local academic research application defined by
[PRD.md](./PRD.md) and its implemented source structure. A `DESIGN.md` file was
not present when this architecture was prepared; the existing interface
implementation and PRD are used for frontend boundaries.

The system has three separately maintained parts:

1. **Frontend:** React, TypeScript, Tailwind CSS, Axios, and Recharts.
2. **Backend:** FastAPI and Pydantic for local HTTP APIs and image handling.
3. **Machine-learning package:** PyTorch and torchvision for offline training,
   evaluation, inference, and Grad-CAM.

The web application does not train models. The dataset, checkpoints, and
evaluation artifacts stay on the local machine and are not committed to Git.
There is no account system, database, remote inference service, or persistent
upload/prediction history. This is not a clinical diagnostic system.

## 2. High-level system architecture

```text
┌────────────────────── Browser ───────────────────────────┐
│ React + TypeScript + Tailwind                            │
│ Upload / preview → research result + Grad-CAM → metrics  │
└───────────────────────────┬─────────────────────────────┘
                            │ Axios
                 multipart/form-data + JSON
                            ▼
┌────────────────────── FastAPI ────────────────────────────┐
│ Routes → schemas → image validation → inference service  │
│                                      ├─ PyTorch model     │
│                                      └─ Grad-CAM          │
│ Metrics route → read real local evaluation report        │
└───────────────────────────┬─────────────────────────────┘
                            │ local filesystem only
                 ┌──────────┴────────────┐
                 ▼                       ▼
       Local model checkpoint     Evaluation report JSON
                 ▲
                 │ offline workflow
┌────────────────┴─────────────────────────────────────────┐
│ Python ML package                                        │
│ Dataset metadata → split → preprocess → train → evaluate │
│ ResNet50 / EfficientNet-B0 → checkpoint and reports       │
└───────────────────────────────────────────────────────────┘
```

The offline workflow and online request path share the model constructors and
versioned preprocessing code, but the web process never invokes the training
CLI.

## 3. Frontend architecture

The frontend is a single dashboard, consistent with the PRD:

- `App.tsx` loads health and evaluation-report state and composes the panels.
- `UploadPanel.tsx` handles file selection, local preview, format/size guidance,
  submit state, and accessible errors.
- `ResultPanel.tsx` presents the predicted class and model score, original
  image, Grad-CAM heatmap, overlay, legend, and research disclaimer.
- `MetricsPanel.tsx` displays metrics and charts only when the API returns a
  real evaluation report.
- `api/client.ts` owns Axios requests and API error extraction.
- `types/api.ts` defines the frontend API contracts.
- `styles.css` and Tailwind utility classes provide shared responsive styling.

The selected image is previewed locally. It is sent only after the user
submits. The frontend does not persist the upload or prediction history.

## 4. Backend architecture

FastAPI separates transport, validation, configuration, and model operations:

- `api/routes/health.py` reports API status and model availability separately.
- `api/routes/predictions.py` exposes image validation and prediction endpoints.
- `api/routes/metrics.py` reads the configured JSON evaluation report.
- `schemas/api.py` defines Pydantic response contracts.
- `services/image_service.py` validates byte size, actual image format, decoded
  dimensions, and malformed content.
- `services/inference_service.py` loads and validates the checkpoint, runs
  classification, and optionally generates Grad-CAM visuals.
- `core/config.py` resolves local model/report paths and upload-size limits.
- `main.py` mounts the API and serves the production frontend build when
  available, otherwise the minimal static fallback.

Model inference runs in a worker thread so synchronous PyTorch work does not
block the FastAPI event loop. Inference and Grad-CAM access to the shared model
are serialized within the process.

## 5. AI model architecture

### Classification models

- **Primary:** torchvision ResNet50 with a two-output classifier for
  `normal` and `glaucoma`.
- **Comparison model:** torchvision EfficientNet-B0 with the same class order.
- **Transfer learning:** torchvision ImageNet default weights are used for new
  training by default; `--no-pretrained` opts out. The current training
  strategy fine-tunes the complete backbone and classifier.
- **Inference:** loads only a local checkpoint whose architecture, class order,
  image size, and preprocessing version match the application.
- **Device:** selects Apple Silicon MPS when available, then CUDA, then CPU.
  No external model service is called.

### Preprocessing and evaluation

Training and inference use shared RGB decoding, EXIF orientation correction,
aspect-ratio-preserving resize/padding to 224 × 224, and ImageNet
normalization. Augmentation is training-only. The train command records a
configuration, split manifest, training history, best checkpoint, and
evaluation outputs in a per-run directory.

If a complete set of REFUGE official train/validation/test partitions is
provided, the splitter preserves them. Otherwise it uses deterministic
stratified image-level splitting or patient-group splitting when complete,
genuine patient IDs are present. The inspected REFUGE metadata did not establish
patient IDs; do not claim those splits are patient-wise.

ResNet50 Grad-CAM targets `layer4[-1].conv3`; EfficientNet-B0 targets its final
feature block. Grad-CAM is an approximate, predicted-class-specific model
explanation. It does not prove glaucoma or identify clinically meaningful
pathology.

Evaluation reports include accuracy, precision, recall/sensitivity,
specificity, F1, ROC-AUC when both classes are present, confusion matrix, ROC
curve, and training/validation curves when available. Never invent or fill in
missing metrics. The held-out test partition is for final evaluation and must
not be used for model or threshold selection.

## 6. API communication flow

1. The browser validates the selected file's declared MIME type and size for
   immediate feedback and displays a local preview.
2. On submit, Axios sends the file as `multipart/form-data` to FastAPI.
3. The backend checks the actual bytes, format, file size, and decoded pixel
   dimensions; browser-supplied names and MIME types are not trusted.
4. For validation-only requests, FastAPI returns format and dimensions; this
   does not verify retinal content or image gradability.
5. For prediction, FastAPI applies shared preprocessing, runs the configured
   checkpoint, and attempts Grad-CAM for the predicted class.
6. The JSON response contains prediction, score, model name, disclaimers, and
   bounded-size PNG data URLs for the original, heatmap, and overlay. If
   explanation generation fails, it returns an explicit explanation error
   without fabricating a visualization.
7. The frontend displays the response. Metrics are fetched separately from the
   configured local report.

## 7. Data flow and component relationships

```text
Offline research
---------------
Read-only source data
  → inspected metadata / labeled manifest
  → official or reproducible split
  → shared preprocessing
  → ResNet50 or EfficientNet-B0 training
  → best checkpoint + config + history + held-out evaluation

Online prediction
-----------------
User file → browser preview → POST /api/predict
  → server-side image validation
  → shared preprocessing → local checkpoint → class scores
  → predicted-class Grad-CAM → heatmap and overlay
  → Pydantic response → React result panel

Metrics
-------
Evaluation report JSON → GET /api/metrics → typed response → Recharts panel
```

```text
App
├── UploadPanel ── api/client.ts ── predictions route
└── ResultPanel ◀── PredictionResponse
    ├── class and model score
    ├── original / Grad-CAM / overlay
    └── explanation and medical-safety notices

App ── api/client.ts ── health route
App ── api/client.ts ── metrics route ◀── local report JSON

predictions route
├── image_service
└── inference_service
    ├── checkpoint / classifier
    ├── shared preprocessing
    └── explainability/gradcam

training CLI
├── data manifest / split / dataset loader
├── preprocessing
├── model constructors
├── training engine
└── evaluation report writer
```

## 8. Project folder structure

The following is the current code organization; runtime data and model folders
are intentionally local and may be empty in a fresh clone.

```text
glaucoma-screening/
├── backend/
│   └── app/
│       ├── api/routes/          # health, image validation/prediction, metrics
│       ├── core/                # local environment configuration
│       ├── schemas/             # Pydantic API contracts
│       ├── services/            # image checks and inference
│       ├── static/              # fallback HTML
│       └── main.py              # FastAPI assembly and static hosting
├── frontend/
│   ├── src/
│   │   ├── api/                 # Axios client
│   │   ├── components/          # upload, result, metrics panels and tests
│   │   ├── types/               # TypeScript API contracts
│   │   ├── App.tsx
│   │   └── styles.css
│   ├── package.json
│   └── vite.config.ts
├── src/glaucoma_ml/
│   ├── data/                    # manifests, splits, dataset, preprocessing
│   ├── explainability/          # reusable Grad-CAM implementation
│   ├── models/                  # ResNet50 and EfficientNet-B0 constructors
│   ├── training/                # training engine and CLI
│   └── evaluation/              # metrics, plots, checkpoint evaluation
├── data/
│   ├── raw/                     # source dataset files; ignored by Git
│   └── processed/               # metadata/manifests; ignored by Git
├── models/                      # local checkpoint files; ignored by Git
├── reports/                     # actual evaluation reports and plots
├── runs/                        # per-experiment outputs; ignored by Git
├── tests/                       # backend and ML tests
├── PRD.md
├── DATASET_PREPARATION_PLAN.md
├── ARCHITECTURE.md
├── README.md
├── requirements.txt
└── .env.example
```

Responsibilities are deliberately split by execution boundary: React concerns
stay under `frontend/`, HTTP concerns under `backend/app/`, and reusable
machine-learning code under `src/glaucoma_ml/`. Dataset and preprocessing
utilities share the `data/` package because the loader and transform contract
are closely coupled; there is no separate duplicate preprocessing package.

## 9. Configuration and environment variables

`backend/app/core/config.py` reads a local `.env` file when present. `.env` is
ignored by Git; `.env.example` contains safe local defaults.

| Variable | Default | Purpose |
|---|---|---|
| `GLAUCOMA_MODEL_PATH` | `models/best_model.pt` | Local trained checkpoint |
| `GLAUCOMA_METRICS_PATH` | `reports/metrics.json` | Local evaluation report read by the dashboard |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum accepted upload size in bytes |

Training hyperparameters, selected architecture, seed, split policy, class
weights, device, and software versions are stored in the experiment
configuration, not in web-service environment variables. Do not store secrets
or dataset credentials in source control.

## 10. Model, dataset, and artifact storage

- Treat files in `data/raw/` as immutable originals. Keep derived metadata and
  split manifests under `data/processed/`.
- Store trained checkpoints locally under `models/` or in their experiment
  directory under `runs/`.
- Store evaluation JSON and generated plots under `reports/` or their run
  directory.
- Git ignores local datasets, checkpoints, run outputs, virtual environments,
  and frontend build/dependency artifacts. GitHub stores source and
  documentation, not the REFUGE images or model weights.
- A checkpoint records its architecture, class order, image size, preprocessing
  version, dataset identity, and training configuration. Inference rejects an
  incompatible checkpoint.
- Verify the exact dataset release, upstream provenance, and usage terms before
  training or sharing derived weights/reports.

There is no database or object store. Prediction uploads and predictions are
not written to disk by the request workflow.

## 11. Error handling and privacy boundaries

- Malformed, empty, unsupported, oversized, or excessive-dimension image files
  receive explicit client errors.
- A missing or incompatible checkpoint makes predictions unavailable; no
  random/untrained fallback model is used.
- Corrupt or invalid evaluation JSON produces an explicit server error rather
  than an empty success or fabricated metrics.
- Grad-CAM failure is logged server-side and reported explicitly while
  preserving an otherwise valid prediction.
- The frontend distinguishes API unavailable, model unavailable, invalid
  upload, prediction failure, and visualization unavailable states.
- Do not log image bytes, retain uploads, or expose local filesystem paths in
  successful responses.

## 12. Deployment architecture

### Development

Run FastAPI/Uvicorn and the Vite development server locally in separate
terminals. Vite proxies `/api` requests to FastAPI. Use the project virtual
environment and local Node dependencies.

### Local production-style run

Build the React application with Vite. FastAPI serves `frontend/dist` static
assets and the API from the same local origin. The local checkpoint and report
remain on the same machine; no remote service or database is required.

### Source control

Use Git and GitHub for source history and collaboration. Do not commit raw or
processed dataset images, local `.env`, checkpoints, or run outputs. Do not
expose the development service publicly. Any hosted or patient-facing use
requires separate privacy, security, licensing, clinical, and regulatory
review and is outside this project's scope.

## 13. Verification and change boundaries

Keep API route schemas, frontend TypeScript contracts, UI states, documentation,
and tests in sync when changing a request or response. The test suite covers
API behavior, image preprocessing, splits, report generation, Grad-CAM, and
frontend rendering. Passing software tests does not establish clinical
validity.

Architecture changes should preserve the local-first, single-dashboard,
no-login, no-database product scope unless the PRD is explicitly revised.
