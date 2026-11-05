# Product Requirements Document

## Document purpose

This document is the source of truth for the functional and non-functional
requirements of the Glaucoma Screening Research Dashboard. It defines a
two-member academic research prototype, not a clinical product. If an
implementation choice conflicts with this document, update and review the
requirement before changing product behavior.

Requirements are labeled:

- **MVP** — required for the local dashboard and its current safe behavior.
- **Research phase** — required only after a dataset has been selected and a
  model has been trained and evaluated.
- **Future** — not required for the academic project.

The app must never imply that a research model confirms or excludes glaucoma.

## 1. Project Overview

**Project title:** Deep Learning-Based Automated Detection System for
Glaucoma Disease

**Product name:** Glaucoma Screening Research Dashboard

The product is a local web application for demonstrating the workflow of
retinal fundus image file validation, research-model inference (only when a
compatible model is available), and review of genuine model-evaluation
artifacts. The user interface is a React dashboard and the local API is
implemented in FastAPI. The offline research pipeline uses Python and PyTorch.

The current deliverable is an academic, single-user prototype with no
accounts, database, image history, cloud service, dataset, or trained
checkpoint. Without a compatible checkpoint, predictions remain unavailable.
The application is not clinically validated and is not intended for care
decisions.

## 2. Problem Statement

Glaucoma can cause progressive vision loss, and research into retinal fundus
image classification can help students understand medical-image machine
learning. A small academic project needs an understandable way to connect
image handling, model development, evaluation, and a web interface.

The prototype addresses the educational workflow, not an unmet clinical
screening need. A fundus photograph and binary dataset labels alone do not
establish a diagnosis. Dataset bias, image quality, label uncertainty, patient
overlap, and performance on populations outside the dataset can invalidate
apparent results. Therefore, the app must communicate uncertainty and
limitations rather than present itself as a diagnostic authority.

## 3. Project Objectives

1. Provide a clear local dashboard to select and preview a candidate retinal
   fundus image.
2. Verify supported file encoding and dimensions on the server before any
   downstream processing.
3. Keep image handling local and avoid retaining uploaded images or creating
   user/prediction records.
4. Provide a modular, reproducible research path for training ResNet50 and,
   optionally, comparing EfficientNet-B0 using a manually selected dataset.
5. Prevent patient leakage where patient identifiers are available and report
   evaluation only from a held-out test partition.
6. Show a prediction only when a compatible locally configured checkpoint is
   available; distinguish a model score from a calibrated probability or
   diagnosis.
7. Make genuine evaluation metrics and plots available without inventing,
   hard-coding, or implying results that have not been generated.
8. Keep setup and verification approachable for two student developers on
   macOS using Python 3.11.

**Success is defined by a reliable, tested academic workflow and honest
reporting—not by a target accuracy or clinical readiness claim.**

## 4. Proposed Solution

The browser sends an image as multipart form data to the local FastAPI
application. The API verifies actual image content, allowed format, upload
size, and decoded dimensions. Validation is not a determination that the
image depicts a retina or is suitable for grading.

When no model is configured, the dashboard stops at validation and clearly
states that no prediction was made. In the research phase, a compatible
checkpoint may enable binary model inference. A separate offline pipeline
loads a manually prepared manifest, creates leakage-aware data partitions,
trains a selected architecture, evaluates it on held-out data, and writes
local reports for the dashboard.

All components run locally by default. There is no remote inference service,
database, login, or persistent upload store.

## 5. Target Users

### Primary

- Student developer/researcher learning computer vision and responsible
  medical-AI prototyping.
- Academic project team members running the application and research pipeline
  locally.

### Secondary

- Instructor, evaluator, or peer reviewing the software workflow and reported
  experiment artifacts.
- A technically curious demonstration user, only when clearly informed that
  the app is not a diagnostic tool.

### Not target users

- Patients seeking a diagnosis, reassurance, triage, or treatment advice.
- Clinicians relying on the output to make or change a care decision.
- Organizations seeking a validated screening or diagnostic system.

## 6. User Personas

### Persona A — Student operator

- **Goal:** Run the project, validate an example fundus image, and understand
  how a model and evaluation report would connect to the dashboard.
- **Needs:** Beginner-friendly setup steps, clear system status, useful input
  errors, and truthful no-model behavior.
- **Risk:** Mistaking a file validation success or model score for clinical
  interpretation.

### Persona B — Student ML developer

- **Goal:** Prepare a lawful dataset, train a baseline model, and examine its
  held-out performance.
- **Needs:** A documented manifest format, reproducible settings, patient-wise
  splitting when possible, and evaluation artifacts with dataset/split
  provenance.
- **Risk:** Data leakage, class imbalance, label ambiguity, or reporting
  internal test metrics as clinical generalization.

### Persona C — Academic reviewer

- **Goal:** Assess whether the implementation is reproducible and whether its
  claims match the evidence.
- **Needs:** Architecture, API contracts, tests, dataset provenance, and
  plainly labeled limitations.
- **Risk:** Treating benchmark performance as external or clinical validation.

## 7. Functional Requirements

### Dashboard and upload

- **FR-01 (MVP):** Show the project purpose, supported input type, local
  processing/privacy statement, and prominent non-diagnostic warning.
- **FR-02 (MVP):** Allow a user to select a JPEG or PNG image and preview it
  locally before sending it to the API.
- **FR-03 (MVP):** Provide an explicit action to validate the selected image.
- **FR-04 (MVP):** Accept only actual supported image encodings, irrespective
  of the filename extension or browser-provided MIME type.
- **FR-05 (MVP):** Reject empty, malformed, unsupported, oversized, or
  excessive-dimension uploads with a clear error and recovery guidance.
- **FR-06 (MVP):** Explain that passing file checks does not verify retinal
  content, image quality, or suitability for classification.
- **FR-07 (MVP):** Remove the local preview when the user replaces or clears
  the selected image; do not retain the uploaded file after request handling.

### Prediction and model status

- **FR-08 (MVP):** Report application availability separately from whether
  predictions are available.
- **FR-09 (MVP):** When a compatible checkpoint is absent or unusable, present
  prediction as unavailable and do not display a class, score, or success-
  shaped fallback.
- **FR-10 (Research phase):** For a compatible model, accept a validated
  fundus-image file and return exactly one of the configured binary class
  labels (`glaucoma` or `normal`), a model score, model identifier, and
  non-diagnostic disclaimer.
- **FR-11 (Research phase):** Label output as an experimental model result;
  never call it a diagnosis, risk assessment, or calibrated probability unless
  calibration has separately been established and documented.
- **FR-12 (Research phase):** Do not infer or display a result if inference
  fails, model metadata is incompatible, or model availability is false.

### Evaluation and research workflow

- **FR-13 (MVP):** Show an explicit unavailable state when no real evaluation
  report exists.
- **FR-14 (Research phase):** Display metrics and plots only when loaded from a
  valid locally generated evaluation report.
- **FR-15 (Research phase):** Provide a documented CSV manifest workflow and
  report whether data partitioning was patient-wise or image-wise.
- **FR-16 (Research phase):** Save the selected model checkpoint and report
  locally; do not commit dataset contents or model weights.
- **FR-17 (MVP):** Provide local API documentation through FastAPI's generated
  OpenAPI/Swagger UI.

## 8. Non-Functional Requirements

### Usability

- **NFR-01:** Setup, run, test, and training instructions must be available in
  the project README and be reproducible by a beginner following them from the
  project directory.
- **NFR-02:** Loading, unavailable, validation-error, inference-error, and
  success states must be distinguishable and use plain language.
- **NFR-03:** The interface must be usable on common desktop, tablet, and
  mobile viewport sizes without hiding core upload or result information.

### Reliability and maintainability

- **NFR-04:** Keep the UI, API routes, configuration, image handling,
  inference, training, and evaluation code in separate modules.
- **NFR-05:** Use explicit configuration for local checkpoint/report paths and
  upload-size limits. Invalid configuration or corrupt artifacts must produce
  an observable error, not a fabricated success.
- **NFR-06:** Cover upload/API behavior, unavailable model/report states,
  evaluation calculations, and split leakage behavior with automated tests.
- **NFR-07:** Results must be reproducible from documented dataset manifest,
  split policy, model architecture, preprocessing, and training configuration
  to the extent supported by the local hardware and framework.

### Performance and platform

- **NFR-08:** The app must run locally on macOS with Python 3.11 and a
  supported Node.js/npm installation.
- **NFR-09:** CPU must be a supported inference/training fallback. Apple
  Silicon MPS may be used when available; the app must not require CUDA.
- **NFR-10:** Image input limits must bound upload bytes and decoded pixel
  dimensions to reduce memory-exhaustion risk. Limits must be configurable or
  documented and enforced on the server.
- **NFR-11:** Model training time is hardware/dataset dependent; the project
  does not promise a fixed training duration or real-time inference service.

### Accessibility

- **NFR-12:** All core controls must be keyboard operable and have accessible
  names, labels, and focus indication.
- **NFR-13:** Text, status messages, and controls must maintain readable
  contrast and must not rely on color alone to convey class, status, or errors.
- **NFR-14:** Dynamic validation and result feedback must be exposed to
  assistive technologies using appropriate semantic/live regions.

## 9. Core Features

1. **Research dashboard:** Single local dashboard with application/model
   status and clear project limitation text.
2. **Image selection and preview:** JPEG/PNG selection with a browser-local
   preview.
3. **Server-side image validation:** Format, content, byte-size, and decoded
   dimension checks.
4. **No-model safe state:** Prediction unavailable until a compatible local
   checkpoint is configured.
5. **Research inference (gated):** Binary output with model score and
   provenance/context, enabled only after a compatible model is available.
6. **Offline ML pipeline:** Manifest-based loading, patient-aware splitting,
   ResNet50 baseline, optional EfficientNet-B0 comparison, training, and
   checkpoint output.
7. **Evaluation dashboard (gated):** Real held-out metrics and plots loaded
   from a local report; otherwise explicit unavailable messaging.

No login, user profiles, database, saved history, sharing, or cloud
deployment is needed for the project's single-user local scope.

## 10. User Workflows

### Workflow A — Start and inspect the dashboard (MVP)

1. The developer installs dependencies and starts FastAPI and the React
   development server, or builds the frontend for local production serving.
2. The user opens the dashboard.
3. The dashboard loads application/model status and metrics availability.
4. The user sees the supported input, local handling statement, and medical
   disclaimer before uploading.
5. If the API is unavailable, the UI shows an actionable service error rather
   than assuming success.

### Workflow B — Validate an image (MVP)

1. The user selects a candidate retinal fundus JPEG or PNG.
2. The browser shows a preview and allows the user to replace or clear it.
3. The user explicitly submits the image for validation.
4. The API validates the uploaded bytes, format, size, and decoded dimensions.
5. The dashboard shows validation success only as a file-format result and
   states that fundus content is not verified.
6. The uploaded file is not retained as history or stored in a database.
7. If no model is available, no prediction is made.

### Workflow C — Research inference (Research phase)

1. The operator has selected and documented an appropriate dataset, trained
   and evaluated a model, and configured a compatible checkpoint and
   preprocessing/class metadata.
2. The dashboard indicates model availability.
3. The user uploads a candidate fundus image.
4. The API repeats server-side validation and runs inference using the
   configured pipeline.
5. The UI presents the predicted class and score as an experimental,
   non-diagnostic model output, along with model identity and limitations.
6. If the checkpoint, metadata, or inference is unavailable, the UI reports
   the error/unavailable state and does not show a class or score.

### Workflow D — Train and evaluate (Research phase)

1. The student manually chooses a public dataset and reads its current terms,
   license, citation, and permitted use.
2. The student records dataset provenance, labels, patient identifier
   availability, and any restrictions; places data locally without committing
   it.
3. The student creates and validates a manifest with image path and binary
   label, plus patient ID when available.
4. The pipeline creates train/validation/test partitions, grouping by patient
   whenever complete IDs permit it; otherwise it records image-level
   partitioning as a limitation.
5. The student trains ResNet50 and may run EfficientNet-B0 as a comparison.
6. The pipeline saves the checkpoint and genuine held-out report locally.
7. The student reviews class-specific metrics, confusion matrix, plots,
   provenance, and limitations before enabling or presenting inference.

## 11. AI/ML Requirements

- **ML-01:** The primary architecture is torchvision ResNet50 adapted for two
  output classes. EfficientNet-B0 is an optional comparison, not a required
  deployment model.
- **ML-02:** The default research training may initialize from torchvision
  ImageNet pretrained weights when the user explicitly runs training and has
  network access. The workflow must document that this may download weights.
  Training from scratch must remain possible.
- **ML-03:** The model class order must be explicit and consistent across
  training, checkpoint metadata, inference, and UI mapping. The required
  labels are `normal` and `glaucoma`.
- **ML-04:** Training and inference must use the same documented image
  decoding, resize/crop, channel, normalization, and tensor conversion
  pipeline. Pillow, torchvision, OpenCV, and NumPy may be used where
  appropriate; avoid undocumented preprocessing differences.
- **ML-05:** Augmentations must be applied only to training data and must not
  change the semantic class. Validation/test transformations must be
  deterministic.
- **ML-06:** Data partitioning must occur at patient level if complete patient
  identifiers are available. If identifiers are absent/incomplete, use a
  stratified image-level split if feasible and clearly record that leakage
  cannot be ruled out.
- **ML-07:** Test data must not be used for model selection, threshold tuning,
  or training. Any threshold selection must use training/validation data and
  document the selected rule.
- **ML-08:** Store sufficient checkpoint metadata to identify architecture,
  class order, image preprocessing, training configuration, and relevant
  dataset/split references. Reject incompatible or incomplete checkpoints.
- **ML-09:** The inference score must be named and described as a model score
  (for example, softmax confidence) unless calibration has been separately
  measured. It must not be presented as a probability of disease.
- **ML-10:** The app must use MPS when supported by the installed PyTorch
  runtime and otherwise use CPU. CUDA is not required for this macOS project.
- **ML-11:** No output may be hard-coded or fabricated to make a demo appear
  functional. No model is included until it is genuinely trained and its
  provenance documented.

## 12. Dataset Requirements

- **DATA-01:** Dataset selection is a manual project decision. The app must
  not automatically fetch datasets.
- **DATA-02:** Before use, students must inspect and record dataset source,
  license/terms, permitted academic/research use, citation, label definitions,
  population, acquisition context, and restrictions. Dataset terms may change;
  verify the current source.
- **DATA-03:** Images must be retinal fundus photographs suitable for the
  defined research task. A dataset of external eye photographs or unrelated
  modalities is not suitable for this classifier.
- **DATA-04:** The manifest must include a path and one binary label per
  image. Labels must map unambiguously to `normal` or `glaucoma`; ambiguous or
  conflicting labels must be excluded or explicitly resolved and documented.
- **DATA-05:** Include patient identifiers when available and permitted.
  Identifiers must be treated as sensitive metadata and must not be surfaced
  in the UI or committed to source control.
- **DATA-06:** Keep raw/processed data and dataset-specific manifests outside
  Git. Never redistribute dataset contents unless its terms explicitly allow
  that use.
- **DATA-07:** Document class counts, excluded/corrupt files, split method,
  and class representation in each split. Fail or clearly warn when a
  meaningful three-way split cannot be made.
- **DATA-08:** Do not use personal or patient images without appropriate
  consent, authorization, privacy safeguards, and an approved research basis.
- **DATA-09:** Dataset labels are research targets, not proof of clinical
  diagnosis. Report the dataset's reference standard and known limitations.

## 13. Frontend Requirements

- **FE-01:** Use React with TypeScript, Vite, Tailwind CSS, Axios, and
  Recharts, consistent with the existing implementation.
- **FE-02:** Provide one dashboard; separate pages, authentication, and complex
  navigation are not required.
- **FE-03:** Show app availability, model/prediction availability, and
  evaluation-report availability as distinct states.
- **FE-04:** Provide a labeled file picker, accepted-format guidance, local
  image preview, replace/clear behavior, and explicit validation/prediction
  actions appropriate to model availability.
- **FE-05:** Distinguish validation feedback from prediction output. A
  validation success must not appear to mean the image is retinal, gradable,
  or classified.
- **FE-06:** Show loading/progress feedback during requests and recoverable,
  understandable errors for failed requests.
- **FE-07:** When a model is available, place the output and disclaimer
  together; do not use reassuring or alarming diagnosis language.
- **FE-08:** Render charts and numeric metrics only for a valid available
  evaluation report. Empty/unavailable state must not contain placeholder
  metrics.
- **FE-09:** Support keyboard navigation, visible focus, semantic labels,
  responsive layout, and status announcements.
- **FE-10:** Do not persist the image in browser storage, analytics, URL, or
  application-managed history.

## 14. Backend Requirements

- **BE-01:** Use FastAPI, Pydantic response schemas, and Uvicorn.
- **BE-02:** Keep health, image validation/prediction, and metrics routes
  modular and keep model/training code separate from web routes.
- **BE-03:** Inspect uploaded bytes rather than trusting filename or supplied
  content type; enforce supported JPEG/PNG format, byte-size, and decoded
  pixel limits.
- **BE-04:** Decode and preprocess images safely; reject malformed inputs with
  a documented client error and a useful message.
- **BE-05:** Do not persist uploads, user identifiers, request bodies, or
  predictions. Avoid logging image bytes or sensitive metadata.
- **BE-06:** Load only a configured local checkpoint that passes compatibility
  checks. Keep checkpoints outside source control.
- **BE-07:** Return explicit unavailable behavior when the checkpoint is
  missing or incompatible; do not silently substitute a random or untrained
  model.
- **BE-08:** Read evaluation metrics from a real local report and return an
  explicit unavailable state when the report does not exist. Invalid reports
  must not be treated as successful results.
- **BE-09:** Use environment-based settings for model/report paths and upload
  limits; provide safe local defaults and document them.
- **BE-10:** The local app must not require or create a database.

## 15. API Requirements

The API is local-first and served under `/api`. Request and response JSON
shapes are validated through Pydantic where applicable. File requests use
`multipart/form-data` with a required `file` field.

### `GET /api/health`

- **Purpose:** Report application health and model availability separately.
- **Success:** HTTP 200 with application status, `predictions_available`, and
  a human-readable model status message.
- **Constraint:** `status: ok` must not imply a trained model or clinically
  valid result.

### `POST /api/images/validate`

- **Purpose:** Check actual file format and safe dimensions without inference.
- **Input:** One JPEG or PNG upload in the `file` field.
- **Success:** HTTP 200 with detected format, width, height, and an explicit
  message that fundus content is not verified and no prediction was made.
- **Errors:** Appropriate 4xx responses for missing, empty, unsupported,
  malformed, oversized, or excessive-dimension content.
- **Constraint:** Must not retain the file after the request.

### `POST /api/predict`

- **Purpose:** Run research inference using a compatible configured local
  model.
- **Input:** One validated image upload in the `file` field.
- **Success (Research phase only):** HTTP 200 with prediction label, model
  score, model identity, and disclaimer.
- **Unavailable:** HTTP 503 when no compatible model is configured or
  inference cannot be served. The UI must not display a prediction for this
  response.
- **Errors:** Invalid uploads use the same validation errors as the validate
  endpoint.
- **Constraint:** The response is experimental and not a diagnosis.

### `GET /api/metrics`

- **Purpose:** Read the latest local evaluation report.
- **Success with report:** HTTP 200 with `available: true`, message, and
  metrics from the report.
- **Success without report:** HTTP 200 with `available: false`, an
  explanatory message, and no fabricated metrics.
- **Invalid report:** Non-success response with an explicit error; never
  convert unreadable or malformed content into an empty successful result.

All API changes must update the Pydantic schemas, frontend client/types,
README, and relevant tests together.

## 16. Model Evaluation Requirements

- Evaluate the selected model on an untouched held-out test split. Do not
  report training-set metrics as test performance.
- Report the dataset name/source and reference, label mapping, split
  strategy, patient-wise versus image-wise status, split counts, architecture,
  checkpoint identity, preprocessing, and evaluation date alongside metrics.
- Report, where computable:
  - confusion matrix with class order;
  - accuracy;
  - per-class precision and recall;
  - glaucoma sensitivity/recall;
  - specificity for the defined negative class;
  - F1 score;
  - ROC curve and ROC-AUC where both classes and valid scores are available;
  - training/validation loss and accuracy curves when recorded.
- State the positive class and averaging convention for every aggregate
  metric. Avoid presenting a single accuracy value without class balance and
  confusion context.
- Handle undefined metrics explicitly (for example, a test split missing a
  class); do not replace them with a favorable numeric value.
- Do not claim that a threshold is clinically appropriate. If a non-default
  threshold is explored, select it using training/validation data and document
  the method before test evaluation.
- Plot and dashboard values must be derived from the same genuine evaluation
  report. No demo, placeholder, or manually invented scores.
- Internal held-out results indicate performance only on that evaluation
  setup. External validation and clinical validation are distinct and are not
  achieved by this academic project.

## 17. Security and Privacy Requirements

- **SEC-01:** Keep the app bound to local development use by default. Do not
  expose the development server to the public internet.
- **SEC-02:** Enforce upload size and decoded-dimension limits server-side;
  reject unsupported or malformed files before inference.
- **SEC-03:** Treat client filenames and MIME types as untrusted. Do not use a
  submitted filename as an unrestricted filesystem path.
- **SEC-04:** Do not save uploads, image previews, prediction history, or
  user-identifying information to a database or persistent application
  storage.
- **SEC-05:** Do not log image contents. Keep logs limited to operational
  errors and non-sensitive status data.
- **SEC-06:** Ignore local `.env`, datasets, model weights, and generated
  artifacts in version control. Never put secrets, private data, or restricted
  dataset files in Git.
- **SEC-07:** Use only datasets and images the team is authorized to process.
  Follow license, terms, consent, and institutional requirements.
- **SEC-08:** Treat model checkpoints and evaluation artifacts as local
  research files; verify their source and compatibility before loading.
- **SEC-09:** A local-only application reduces exposure but does not itself
  guarantee privacy, security, regulatory compliance, or safe clinical use.

## 18. Medical Disclaimer

This application is an academic research prototype for educational
experimentation. It is not a medical device and has not been clinically
validated. It does not diagnose, rule out, screen for, or provide treatment
advice for glaucoma. Any model output is experimental and may be incorrect,
biased, uncalibrated, or inapplicable to a particular image or person. A
negative/normal output must not reassure a user or delay care. Do not use
outputs to make clinical decisions. People with eye-health concerns should
consult a qualified eye-care professional.

Display this notice prominently before upload and alongside every research
prediction. Never remove or weaken it without an explicit product and
qualified clinical/regulatory review.

## 19. Project Scope

### Included

- Single-user local web dashboard with responsive image selection and preview.
- Local FastAPI API for health, image validation, prediction availability,
  optional research inference, and evaluation report access.
- Server-side JPEG/PNG verification and configured upload/dimension limits.
- PyTorch/torchvision research model foundation: ResNet50 baseline and
  optional EfficientNet-B0 comparison.
- Local manifest-based training and held-out evaluation workflow.
- Patient-wise split support when valid identifiers are available.
- Genuine metrics/report generation and dashboard display when available.
- Tests, setup documentation, and prominent non-diagnostic language.

### Scope boundary

The existing application has a dashboard, API validation, no-model behavior,
metrics-unavailable behavior, and a training/evaluation foundation. Model
training, predictions, and real evaluation charts require user-selected data,
completed training/evaluation, and a compatible local checkpoint/report; they
must not be treated as completed merely because the code path exists.

## 20. Out of Scope Features

- Confirmed glaucoma diagnosis, clinical screening recommendation, triage, or
  treatment guidance.
- Clinical deployment, regulatory approval, or claims of clinical
  effectiveness.
- Ordinary eye selfies, anterior-segment photos, or other non-fundus image
  modalities.
- Dataset scraping, automatic dataset acquisition, redistribution, or
  inclusion of dataset images in the repository.
- Accounts, roles, password reset, patient records, database, image history,
  cloud storage, image sharing, or multi-user operation.
- Remote inference, public hosting, mobile native app, or production
  availability guarantees.
- A promise of accuracy, sensitivity, specificity, or real-time response
  before a real evaluation has been completed.
- Automatic diagnosis or image-quality/gradable-fundus detection unless a
  separately scoped and validated research task is approved.

## 21. Future Enhancements

These are not commitments for the two-member academic MVP and require separate
scope and safety review:

- External-dataset evaluation and a documented generalization study.
- A separately trained and evaluated fundus-content or image-quality gate.
- Confidence calibration and uncertainty analysis, with careful user-facing
  interpretation.
- More robust model comparison, repeated seeds, confidence intervals, and
  subgroup analysis where dataset size and lawful metadata allow.
- Improved reproducibility tooling, such as a captured environment and
  experiment configuration.
- Accessibility/usability study with representative academic users.
- Any multi-user, hosted, or clinical workflow only after privacy, security,
  clinical, regulatory, and institutional review.

## 22. Acceptance Criteria

### MVP acceptance — current product foundation

- **AC-01:** A developer can follow the README to install and run the local
  backend and frontend on macOS with Python 3.11.
- **AC-02:** The dashboard visibly identifies the application as an academic
  research prototype and displays the medical disclaimer.
- **AC-03:** A user can select, preview, replace, and clear a supported image.
- **AC-04:** Server validation accepts a valid JPEG/PNG within configured
  limits and reports format/dimensions without claiming retinal-content
  verification.
- **AC-05:** The API rejects missing, malformed, unsupported, oversized, and
  excessive-dimension uploads with clear non-success responses.
- **AC-06:** The health response distinguishes application health from model
  availability.
- **AC-07:** With no compatible checkpoint, prediction is unavailable; no
  label, confidence, or placeholder prediction is shown.
- **AC-08:** Without an evaluation report, metrics are explicitly unavailable
  and no chart values are fabricated.
- **AC-09:** The app does not create user accounts, database records, image
  history, or persistent upload files.
- **AC-10:** Core upload and status flows are keyboard accessible and have
  visible, understandable feedback.
- **AC-11:** Automated backend and frontend tests pass, and the frontend
  production build succeeds.

### Research-phase acceptance — only after dataset/model work

- **AC-12:** The chosen dataset has documented provenance, terms/license,
  citation, label definitions, population, permitted use, and patient-ID
  availability before training.
- **AC-13:** The manifest is validated; invalid/missing images and ambiguous
  labels are rejected or explicitly accounted for.
- **AC-14:** Patient-wise split isolation is tested and used when complete
  patient identifiers are available. Otherwise, image-wise splitting is
  clearly identified as a limitation.
- **AC-15:** Training configuration, architecture, class order, preprocessing,
  split information, and checkpoint metadata are saved and compatible with
  inference.
- **AC-16:** Held-out evaluation is generated from real model outputs and
  exposes metric definitions, class order, split provenance, and unavailable
  metric cases.
- **AC-17:** Dashboard charts match the saved evaluation report and remain
  unavailable when the report is absent or invalid.
- **AC-18:** Prediction is enabled only with a compatible checkpoint and
  returns a binary class, clearly named model score, model identity, and
  disclaimer; errors never fall back to fabricated output.
- **AC-19:** Project materials state that internal metrics and model outputs
  do not constitute external or clinical validation.

## Change Control

Before adding a feature or changing an acceptance criterion, confirm that it
fits the two-member academic scope, preserves the no-login/no-database
approach, does not imply diagnosis, and can be supported by available data and
evidence. Update this document and relevant code/tests/documentation together.
