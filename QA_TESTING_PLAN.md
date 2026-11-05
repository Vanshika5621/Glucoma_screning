# Testing and Quality Assurance Plan

## 1. Purpose and safety boundary

This plan describes quality checks for the local Glaucoma Detection research
application. It covers the React dashboard, FastAPI APIs, image validation,
PyTorch model loading and inference, Grad-CAM, evaluation artifacts, and privacy
boundaries.

The application is an academic research prototype, not a medical device. A
passing software test does not establish diagnostic safety, clinical validity,
or generalization. Test only with authorized research images or synthetic
non-patient images. Do not put identifiable patient images in Postman history,
screenshots, test fixtures, CI logs, Git, or shared reports.

## 2. Test environment and commands

Run Python tests from the project root using the project virtual environment:

```bash
.venv/bin/python -m pytest
```

Run React component tests and a TypeScript/production build:

```bash
cd frontend
npm test -- --run
npm run build
```

Run the local API for manual Postman testing:

```bash
cd ..
source .venv/bin/activate
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Use `http://127.0.0.1:8000` as the Postman base URL. Keep the server bound to
localhost; do not expose it to a public network.

An importable, sanitized collection is available at
[`postman/Glaucoma-Local-API.postman_collection.json`](./postman/Glaucoma-Local-API.postman_collection.json).
Import it into Postman and set the collection variables
`sampleImagePath`, `unsupportedImagePath`, and `malformedImagePath` to local
non-identifying fixtures before running. The collection contains no images or
predetermined predictions. The prediction request requires a compatible local
checkpoint.

### Postman procedure

For `POST /api/images/validate` and `POST /api/predict`:

1. Select **Body → form-data**.
2. Add key `file`, set its type to **File**, and select an authorized,
   non-identifying JPEG or PNG fixture.
3. Do not manually set `Content-Type`; let Postman add the multipart boundary.
4. Save only synthetic or otherwise non-sensitive requests in collections or
   history. Disable request/response body logging or remove saved histories
   when testing sensitive workflows.
5. Confirm the HTTP status, response fields, and absence of unexpected
   filesystem artifacts. Record the observed result and tester/date in the
   Actual Result column below.

For health and metrics, send `GET /api/health` and `GET /api/metrics`.
`GET /docs` exposes the local OpenAPI UI but does not replace the Postman
negative-input cases.

## 3. Status definitions

| Status | Meaning |
|---|---|
| **PASS** | Executed and matched the expected result in the recorded baseline. |
| **FAIL** | Executed and did not match; attach a defect/reference and do not release. |
| **NOT RUN** | Planned test without recorded execution evidence. |
| **PARTIAL** | Some assertions/evidence exist, but the complete scenario remains unverified. |
| **BLOCKED** | Cannot run until a prerequisite is available; state that prerequisite. |

Do not mark a planned manual Postman, browser, privacy, or device test as PASS
until it has actually been run and observed.

## 4. Baseline execution record

Automated baseline recorded **2026-11-05**:

- Python: `.venv/bin/python -m pytest` — **39 passed**. One non-blocking
  Starlette deprecation warning notes that its TestClient currently uses
  `httpx`; no test failed.
- Frontend: `npm test -- --run` — **7 passed**.
- Frontend build: `npm run build` — **passed** (TypeScript check and Vite
  production build).
- Additional automated edge cases: empty upload, content-vs-filename/MIME,
  path-like filename not written to disk, malformed metrics JSON, corrupt and
  incompatible checkpoint rejection, unsupported browser type, and browser
  upload-size rejection — **passed**.
- The Postman collection parses as valid JSON, but neither Postman nor Newman
  is installed/available in this environment; its requests have not been
  executed from Postman.
- `.gitignore` rules were reviewed and updated to ignore all generated report
  contents. Git-based tracked/ignored status cannot be verified because this
  workspace is not a Git repository.
- Browser responsive smoke: core upload/result/metrics sections visible with
  a real prediction/Grad-CAM result and no horizontal overflow at widths
  **320, 375, 768, 1024, 1280, and 1440 px**. Keyboard focus on the upload
  control shows a 3 px outline; reduced-motion preference was also checked in
  browser emulation. This does not replace screen-reader or physical-device
  review.
- Real-checkpoint compatibility: the configured ResNet50 checkpoint loaded on
  MPS and exposed `layer4[-1].conv3` as a Conv2d target layer.
- Real-image API smoke test (not a clinical test metric): one existing REFUGE
  official-training image, `g0001.jpg`, returned HTTP 200, predicted class
  `glaucoma`, model score `0.939`, and PNG data URLs for all three visualizations.
  This single training-partition image is only a feature smoke test. It is not
  held-out performance evidence and is not included in the test-set metrics.
- Manual Postman execution, responsive-device checks, persistence inspection,
  and privacy/security review have **not** been recorded.

## 5. Unit testing strategy

Use pytest for deterministic Python units and Vitest with React Testing Library
for UI components. Keep unit tests fast and isolated from real checkpoint
weights and external downloads. Mock external dependencies only when testing
error branches; never report mocked outputs as real model predictions.

| Test Case ID | Test Scenario | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| UNIT-001 | Decode, orient, RGB convert, resize and pad image | Synthetic RGB image with known dimensions/aspect ratio | 224 × 224 image, RGB, aspect ratio retained with centered padding | Pytest asserts output dimensions and padding pixels | PASS |
| UNIT-002 | Normalize inference transform | Synthetic decoded image | Deterministic float tensor with shape `[3,224,224]` | Pytest asserts shape, type consistency, and deterministic inference transform | PASS |
| UNIT-003 | Apply training-only augmentation transform | Synthetic decoded image | Tensor shape remains `[3,224,224]`; validation/inference transforms remain deterministic | Pytest checks training transform output shape; randomized augmentation distribution not statistically evaluated | PARTIAL |
| UNIT-004 | Compute balanced loss weights | Training labels 6 normal, 2 glaucoma | Weights match inverse-frequency formula, `[2/3, 2]` | Pytest asserts exact expected tensor | PASS |
| UNIT-005 | Reject class weighting when a class is absent | Training frame containing only normal labels | Clear error; no implicit fabricated class | Pytest expects `ValueError` | PASS |
| UNIT-006 | Preserve distinct patient groups | Synthetic grouped manifest | No patient ID overlap among train, validation, and test | Pytest checks disjoint patient sets and both classes | PASS |
| UNIT-007 | Reject incomplete patient identifiers | Manifest with null/blank patient IDs | Fail closed rather than silently using image-level split | Pytest covers missing and blank patient IDs | PASS |
| UNIT-008 | Deterministically repeat patient split | Same synthetic manifest and seed | Identical partition assignment | Pytest compares split paths | PASS |
| UNIT-009 | Preserve official dataset partitions | Manifest with all three official partitions | Train, validation, and test memberships are preserved; partial partitions rejected | Pytest covers complete and partial official partition cases | PASS |
| UNIT-010 | Generate classification report and plots | Explicit labels, scores, and sample training history | Correct measured metrics; confusion, ROC, and accuracy/loss plots written | Pytest verifies metric values and output files | PASS |
| UNIT-011 | Generate class-specific Grad-CAM map | Deterministic tiny test classifier and tensor | Finite normalized map; target layer hook removed after execution | Pytest asserts finite normalized map; hook cleanup not separately asserted | PARTIAL |
| UNIT-012 | Align and encode explanation visuals | Synthetic image and heatmap | Original, heatmap, and overlay are valid same-size PNG data URLs | Pytest decodes all three and checks dimensions | PASS |
| UNIT-013 | Preserve existing experiment files | Non-empty output directory | New run refused; existing content unchanged | Pytest verifies exception and unchanged file | PASS |

## 6. Backend API testing strategy (pytest and Postman)

Automated API tests use FastAPI TestClient and synthetic images. Postman cases
below are manual checks and require their own recorded observation. Test both
the expected HTTP status and the response schema.

| Test Case ID | Test Scenario | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| API-001 | Health with model unavailable | `GET /api/health`; mock no checkpoint | HTTP 200; app `ok`, predictions unavailable, explanatory message | Pytest asserts unavailable model state | PASS |
| API-002 | Health with real compatible checkpoint | `GET /api/health` with configured local checkpoint | HTTP 200; model availability true and model/device reported; not a validity claim | Earlier API check returned ResNet50 available on MPS | PASS |
| API-003 | Metrics unavailable | `GET /api/metrics` with nonexistent report path | HTTP 200; `available:false`, `metrics:null`, no synthetic values | Pytest asserts unavailable/no-metric response | PASS |
| API-004 | Metrics report malformed | `GET /api/metrics` with invalid JSON or invalid report shape | Non-success response with explicit error; no empty-success fallback | Not covered by recorded test baseline | NOT RUN |
| API-005 | Validate supported PNG | `POST /api/images/validate`, valid small PNG | HTTP 200; format/dimensions; says retina not verified and no prediction made | Pytest asserts response and safety text | PASS |
| API-006 | Predict without a compatible checkpoint | `POST /api/predict`, valid PNG, model unavailable | HTTP 503; no prediction or confidence | Pytest asserts no-model error and no placeholder | PASS |
| API-007 | Predict with real checkpoint and Grad-CAM | `POST /api/predict`, authorized REFUGE training image | HTTP 200; valid class/score, disclosures, three PNG data URLs | Real API smoke returned HTTP 200 and all three PNGs | PASS |
| API-008 | Postman happy-path validation | Postman multipart JPEG/PNG | HTTP 200 and valid validation schema | Collection request prepared; Postman/Newman unavailable, so no Postman run recorded | NOT RUN |
| API-009 | Postman happy-path prediction | Postman multipart authorized research image; local checkpoint configured | HTTP 200 and schema includes prediction and explanation fields | Collection request prepared; Postman/Newman unavailable; equivalent direct HTTP smoke was executed | NOT RUN |
| API-010 | Missing multipart file field | Postman request with no `file` field | FastAPI validation error; no inference | No Postman run recorded | NOT RUN |

## 7. Image upload and validation test cases

Server-side checks are authoritative; browser `accept`, filename, and MIME type
are not trusted. Current limits are 10 MiB by default and 40,000,000 decoded
pixels. Tests that exceed limits should use generated fixtures or monkeypatched
limits rather than allocating dangerously large images.

| Test Case ID | Test Scenario | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| IMG-001 | Valid PNG upload | Well-formed PNG under limits | HTTP 200 from validate endpoint | Pytest passes | PASS |
| IMG-002 | Corrupt/non-image bytes | Arbitrary text bytes named `.jpg` | HTTP 400; useful invalid-image message | Pytest passes | PASS |
| IMG-003 | Unsupported but decodable format | BMP image | HTTP 415; only JPEG/PNG accepted | Pytest passes | PASS |
| IMG-004 | Empty upload | Zero bytes | HTTP 400; no inference | Not covered by recorded test baseline | NOT RUN |
| IMG-005 | Upload exceeds configured byte limit | Payload larger than `MAX_UPLOAD_BYTES` | HTTP 413 before decode/inference | Pytest tests a monkeypatched 4-byte limit and rejects 5 bytes | PASS |
| IMG-006 | Decoded dimensions exceed pixel cap | Image with pixel count greater than configured maximum | HTTP 413; decode-bomb warning/error handled | Pytest uses a small monkeypatched cap and verifies 413 | PASS |
| IMG-007 | MIME/extension mismatch | PNG bytes named `.jpg`, misleading content-type | Validation follows decoded bytes/actual format; no trust of client metadata | Pytest verifies PNG bytes are detected as PNG despite path-like `.jpg` filename and wrong MIME; no file is written | PASS |
| IMG-008 | Truncated JPEG/PNG | Incomplete image stream | HTTP 400; no inference or traceback | Not covered by recorded test baseline | NOT RUN |
| IMG-009 | Valid JPEG upload | Well-formed JPEG under limits | HTTP 200 with detected format and dimensions | No direct JPEG API test recorded | NOT RUN |
| IMG-010 | Replace selected image in UI | Select image A then image B | Preview updates; previous object URL released; stale result cleared | Browser interaction test not recorded | NOT RUN |

## 8. Model loading, prediction, and explanation testing

Use the real local checkpoint only for compatibility and smoke tests. Automated
unit tests may use deterministic toy networks for Grad-CAM mechanics, but those
outputs are never presented as glaucoma predictions. Do not assert a particular
class for arbitrary user uploads as a correctness oracle.

| Test Case ID | Test Scenario | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| ML-001 | Load configured ResNet50 checkpoint | Existing `models/best_model.pt` | Strict state load succeeds; architecture/class order/preprocessing match | Real checkpoint loaded on MPS; target `layer4[-1].conv3` resolved | PASS |
| ML-002 | Missing model checkpoint | Nonexistent model path | Model unavailable; prediction returns HTTP 503 | API test exercises unavailable model path with controlled mock | PASS |
| ML-003 | Corrupt/incompatible checkpoint | Invalid checkpoint, class order, image size, or preprocessing version | Model is rejected with observable unavailable state; never random model | No direct corrupt/incompatible checkpoint test recorded | NOT RUN |
| ML-004 | Actual model inference smoke | `g0001.jpg` from official REFUGE training partition | Actual checkpoint produces observed output; output not treated as held-out metric | Observed `glaucoma`, score `0.939`; this is a single training image only | PASS |
| ML-005 | Grad-CAM class score and output shape | Deterministic unit-test network | Target class gradients produce finite normalized map of expected size | Pytest mechanics pass; not a test of clinical interpretability | PASS |
| ML-006 | Correct ResNet50 target layer | Compatible trained ResNet50 | Hook targets final bottleneck `layer4[-1].conv3` | Verified against the loaded checkpoint; target is Conv2d | PASS |
| ML-007 | Explanation generation failure | Model/architecture without expected target layer | Keep valid prediction; return explicit visualization-unavailable state; do not fabricate images | Unit test returns prediction and error with no overlay | PASS |
| ML-008 | Grad-CAM API JSON images | Real model and authorized image | Original, heatmap, overlay are valid PNG data URLs and dimensions align | Direct API smoke verified all three PNG data URLs | PASS |
| ML-009 | EfficientNet-B0 compatibility and Grad-CAM | EfficientNet-B0 checkpoint | Strict load and final feature target work | No trained EfficientNet checkpoint or run recorded | BLOCKED |
| ML-010 | Repeatable real prediction | Same model/build/input/device | Compare numerical scores within a documented tolerance; no cross-hardware bitwise assumption | Repeatability test not recorded | NOT RUN |

## 9. Frontend component and responsive UI testing

Automated component tests cover accessible state/content behavior using
Vitest, jsdom, and React Testing Library. A production build is not a visual
responsive test. Use browser/device emulation and keyboard checks at the
viewports below.

| Test Case ID | Test Scenario | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| UI-001 | Dashboard safety and unavailable states | Render App with unavailable health/report mocks | No fabricated prediction or evaluation metrics; non-diagnostic notice visible | Vitest App tests pass | PASS |
| UI-002 | Validate image UI flow | Upload synthetic browser File; model unavailable | Validation status shown; prediction client not called | Vitest App test passes | PASS |
| UI-003 | Display Grad-CAM results | Typed prediction response with three image data URLs | Predicted class, original, heatmap, overlay, legend, and caution shown | Vitest ResultPanel test passes | PASS |
| UI-004 | Explanation unavailable state | Response has explanation error/null visual fields | Prediction remains visible; clear explanation error shown | Vitest ResultPanel test passes | PASS |
| UI-005 | Wrong browser MIME type | Select file with unsupported MIME type | Immediate clear error; Analyze disabled; no request | Vitest confirms the alert and no API request | PASS |
| UI-006 | Browser upload exceeds 10 MiB | Select oversized File | Immediate clear error; no request | Vitest confirms the size alert and no API request | PASS |
| UI-007 | API failure/recovery | Make prediction request fail, then retry with a valid response | Error is announced; busy state clears; retry possible; no stale prediction | No direct test recorded | NOT RUN |
| UI-008 | Narrow mobile viewport with result visible | 320 × 640 and 375 × 812 | No horizontal overflow; upload/results remain readable and usable | Real prediction/three Grad-CAM images rendered at 320 px; no horizontal overflow at 320 and 375 px | PASS |
| UI-009 | Tablet viewport | 768 × 1024 | Components reflow without clipped images/forms/charts | Browser viewport run at 768 px with result present: no horizontal overflow; all three explanation images present | PASS |
| UI-010 | Desktop viewport | 1280 × 800 and 1440 × 900 | Two-column dashboard and chart panels readable | Browser viewport runs at 1280 and 1440 px with result present: no horizontal overflow; three explanation images present | PASS |
| UI-011 | Keyboard focus visibility | Tab to upload control and inspect focus style | Core controls keyboard reachable with visible focus indicator | Upload input reached by keyboard; computed outline is solid 3 px. Screen-reader announcement audit not run | PARTIAL |
| UI-012 | Reduced motion | Browser `prefers-reduced-motion: reduce` emulation | Non-essential animations/transitions suppressed | Emulation active; button transition computed as 0.00001 s | PASS |

## 10. Integration and end-to-end testing

| Test Case ID | Test Scenario | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| INT-001 | API validates unsupported upload | TestClient multipart BMP | HTTP 415 and helpful error | Backend automated test passes | PASS |
| INT-002 | API to inference to Grad-CAM | Real local checkpoint and authorized image | HTTP 200 response contains actual prediction and all visualizations | Direct HTTP API smoke returned a real prediction and three visuals | PASS |
| INT-003 | Frontend to backend browser flow | Upload authorized training image in dashboard | Prediction and all Grad-CAM images render; caveat visible | Browser run showed prediction and three images | PASS |
| INT-004 | Metrics route to dashboard | Existing real report configured | Dashboard plots/values derive from report | Browser showed real internal 60-sample report; no fabrication | PASS |
| INT-005 | Model unavailable end-to-end | Stop/configure missing checkpoint and refresh | Status unavailable; predict endpoint returns 503; UI shows no result | Unit/API unavailable branch tested; full browser path not recorded | PARTIAL |
| INT-006 | Invalid metric report end-to-end | Configure malformed JSON and load dashboard | API error and clear UI recovery state; no fabricated chart | Not recorded | NOT RUN |

## 11. Security and privacy testing

Perform privacy checks only with synthetic or authorized non-identifying test
data. Do not use actual patient records as fixtures. Local-only execution is a
boundary, not a privacy or security guarantee.

| Test Case ID | Test Scenario | Input | Expected Result | Actual Result | Status |
|---|---|---|---|---|---|
| SEC-001 | Malicious filename/path | Multipart filename such as `../../sample.png` with valid image bytes | Filename is never used as a destination path; request safely handled | Not directly tested | NOT RUN |
| SEC-002 | Unsupported/malformed payload | BMP, corrupt bytes, oversized image | Rejected before model inference with expected 4xx | Automated tests cover BMP, corrupt bytes, byte and pixel limits | PASS |
| SEC-003 | Check persistence of uploaded file | Record a temporary directory before/after path-like filename upload | No upload is written to disk by image-validation request | Automated pytest changes to temporary working directory and confirms no files are created for path-like filename; prediction persistence not independently checked | PARTIAL |
| SEC-004 | Check logs and error responses | Trigger decode/model errors using synthetic files | No image bytes, local image path, traceback, or identifying metadata in client response/logs | Not formally audited | NOT RUN |
| SEC-005 | Verify Git ignore rules | Inspect status/ignore behavior for `.env`, datasets, weights, reports, runs | Sensitive/local assets ignored; no secret/data committed | `.gitignore` reviewed and generated report outputs are now ignored; Git status/tracked-file audit unavailable because the workspace has no `.git` repository | PARTIAL |
| SEC-006 | Local bind and exposure | Inspect server bind address/process and reachable interfaces | Default/test server only on loopback; no public exposure | Uvicorn local host was used for browser smoke; external-interface scan not run | PARTIAL |
| SEC-007 | Request/body retention in Postman | Use synthetic fixture, inspect Postman history/settings | No identifiable patient data saved in request history or exported collection | No Postman privacy check recorded | NOT RUN |
| SEC-008 | Dependency and secret review | Review dependency advisories, secret scan, and source-control diff | No known vulnerable dependency/secrets; issues tracked for remediation | No dedicated scan recorded | NOT RUN |
| SEC-009 | Response size/resource behavior | Authorized maximum-size valid image | Bounded response; no unbounded image dimensions; request completes or fails safely | Output thumbnails bounded to 1024 px; load/latency test not recorded | PARTIAL |

## 12. Model quality and scientific QA

Software correctness tests and model performance testing are separate:

- Keep training images, validation images, and final test images distinct.
- Patient-wise splits require real patient identifiers. The inspected REFUGE
  metadata did not establish them, so its partitions must not be described as
  patient-wise.
- Do not use official test metrics to choose architecture, epochs, weights,
  class threshold, or preprocessing. Use validation data for those decisions.
- Report the class counts, confusion matrix, sensitivity, specificity,
  precision, recall, F1, ROC-AUC, threshold, checkpoint, preprocessing version,
  source, and split policy with every result.
- Do not present model scores as calibrated probabilities or Grad-CAM as
  localization proof. The model has not been clinically validated.
- Compare ResNet50 and EfficientNet-B0 only on identical manifests/splits and
  evaluation protocol. No EfficientNet-B0 result should be claimed until a
  separate real run is completed and its report exists.

Existing recorded held-out evaluation artifacts are not rerun by this QA plan.
Their displayed performance must be described with their actual data split and
class imbalance; passing the test suite does not increase those metrics or
establish external validation.

## 13. Release gates and defect handling

Before sharing a software change:

1. Python suite passes; frontend tests and TypeScript production build pass.
2. Relevant new/changed API cases have matching Pydantic and TypeScript
   contracts and automated coverage.
3. Invalid uploads fail closed; no-model and invalid-report states show no
   fabricated values.
4. Real-checkpoint smoke test, if applicable, records architecture, device,
   input source partition, and observed outputs without presenting them as
   held-out metrics.
5. No patient data, raw datasets, checkpoints, `.env`, or generated run output
   is added to Git.
6. Manual Postman and responsive checks are recorded only after actual
   execution.
7. All user-facing model outputs retain research-only and Grad-CAM limitations.

Record defects with case ID, reproducible steps, environment, expected versus
actual result, severity, and a sanitized log/screenshot. Never attach a patient
image or a response containing identifying information to a defect report.
