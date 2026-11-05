import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { getErrorMessage, predictImage, validateImage } from "../api/client";
import type { PredictionResponse } from "../types/api";

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png"];

interface UploadPanelProps {
  predictionsAvailable: boolean;
  onPrediction: (result: PredictionResponse | null) => void;
}

export function UploadPanel({
  predictionsAvailable,
  onPrediction,
}: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(
    () => () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    },
    [previewUrl],
  );

  function chooseFile(candidate?: File) {
    setError(null);
    setMessage(null);
    onPrediction(null);
    setFile(null);
    setPreviewUrl(null);
    if (!candidate) return;
    if (!ACCEPTED_TYPES.includes(candidate.type)) {
      setError("Choose a JPEG or PNG retinal fundus image.");
      return;
    }
    if (candidate.size > MAX_FILE_BYTES) {
      setError("The image is larger than 10 MB. Choose a smaller file.");
      return;
    }
    setFile(candidate);
    setPreviewUrl(URL.createObjectURL(candidate));
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      if (predictionsAvailable) {
        onPrediction(await predictImage(file));
      } else {
        setMessage(await validateImage(file));
      }
    } catch (requestError) {
      setError(getErrorMessage(requestError));
      onPrediction(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel p-6 sm:p-8" aria-labelledby="upload-title">
      <div>
        <p className="eyebrow">IMAGE ANALYSIS</p>
        <h2 id="upload-title" className="mt-2 text-xl font-semibold">
          Upload a retinal image
        </h2>
        <p className="mt-2 max-w-2xl text-sm leading-6 text-muted">
          Use a fundus photograph that shows the retina and optic nerve. An
          ordinary selfie of the front of the eye is not supported.
        </p>
      </div>

      <form className="mt-6" onSubmit={submit}>
        <label
          htmlFor="fundus-image"
          className="flex min-h-48 cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-slate-400 bg-slate-50 p-6 text-center hover:bg-slate-100 focus-within:ring-2 focus-within:ring-primary"
        >
          <span className="text-base font-semibold">
            {file ? "Choose a different image" : "Choose an image"}
          </span>
          <span className="mt-2 text-sm text-muted">
            JPEG or PNG · up to 10 MB
          </span>
          <input
            id="fundus-image"
            className="sr-only"
            type="file"
            accept="image/jpeg,image/png"
            onChange={(event) => chooseFile(event.target.files?.[0])}
          />
        </label>

        {previewUrl && (
          <div className="mt-5">
            <img
              className="mx-auto max-h-80 max-w-full rounded-lg border border-line object-contain"
              src={previewUrl}
              alt="Preview of the selected retinal image"
            />
            <p className="mt-2 break-all text-center text-xs text-muted">
              {file?.name}
            </p>
          </div>
        )}

        {!predictionsAvailable && (
          <p className="mt-4 rounded-lg bg-slate-100 p-4 text-sm leading-6 text-muted">
            The upload workflow is ready, but no trained model is configured.
            This image will not be sent for prediction.
          </p>
        )}

        {message && (
          <p
            className="mt-4 rounded-lg bg-emerald-50 p-4 text-sm leading-6 text-emerald-900"
            role="status"
          >
            {message}
          </p>
        )}

        {error && (
          <p
            className="mt-4 rounded-lg bg-red-50 p-4 text-sm text-red-800"
            role="alert"
          >
            {error}
          </p>
        )}

        <button
          className="mt-5 min-h-11 rounded-lg bg-primary px-5 py-2.5 font-semibold text-white hover:bg-teal-900 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary disabled:cursor-not-allowed disabled:opacity-50"
          type="submit"
          disabled={!file || busy}
        >
          {busy
            ? predictionsAvailable
              ? "Analyzing…"
              : "Validating…"
            : predictionsAvailable
              ? "Analyze image"
              : "Validate image (no prediction)"}
        </button>
      </form>
    </section>
  );
}
