import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { getErrorMessage, predictImage } from "../api/client";
import type { PredictionResponse } from "../types/api";

const MAX_FILE_BYTES = 10 * 1024 * 1024;

interface UploadPanelProps {
  predictionsAvailable: boolean;
  onPrediction: (result: PredictionResponse | null, file?: File) => void;
}

export function UploadPanel({
  predictionsAvailable,
  onPrediction,
}: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(
    () => () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    },
    [previewUrl],
  );

  function chooseFile(candidate?: File) {
    setError(null);
    onPrediction(null);
    setFile(null);
    setPreviewUrl(null);
    if (!candidate) return;
    if (candidate.type && !candidate.type.startsWith("image/")) {
      setError("Choose a supported retinal fundus image.");
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
    if (!file || !predictionsAvailable) return;
    setBusy(true);
    setError(null);
    try {
      onPrediction(await predictImage(file), file);
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
        <h2 id="upload-title" className="mt-2 text-xl font-semibold">
          Upload image
        </h2>
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
            Image · up to 10 MB
          </span>
          <input
            id="fundus-image"
            className="sr-only"
            type="file"
            accept="image/*"
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
          disabled={!file || busy || !predictionsAvailable}
        >
          {busy ? "Analyzing…" : "Analyze image"}
        </button>
      </form>
    </section>
  );
}
