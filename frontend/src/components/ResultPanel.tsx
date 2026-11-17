import { useState } from "react";
import type { PredictionResponse } from "../types/api";

export function ResultPanel({
  result,
  modelAvailable,
}: {
  result: PredictionResponse | null;
  modelAvailable: boolean;
}) {
  const [copyMessage, setCopyMessage] = useState<string | null>(null);

  async function copyResult() {
    if (!result) return;
    const prediction = result.prediction === "glaucoma" ? "Yes" : "No";
    const text = `Prediction: ${prediction}\nConfidence: ${(result.confidence * 100).toFixed(1)}%`;
    try {
      await navigator.clipboard.writeText(text);
      setCopyMessage("Result copied.");
    } catch {
      setCopyMessage("Could not copy the result. Check clipboard permissions.");
    }
  }

  return (
    <section className="panel p-6 sm:p-8" aria-labelledby="result-title">
      <h2 id="result-title" className="mt-2 text-xl font-semibold">
        Prediction
      </h2>
      {result ? (
        <div className="mt-6" role="status" aria-live="polite">
          <p className="text-4xl font-bold">
            {result.prediction === "glaucoma" ? "Yes" : "No"}
          </p>
          <p className="mt-4 text-base">
            Confidence:{" "}
            <span className="font-semibold">
              {(result.confidence * 100).toFixed(1)}%
            </span>
          </p>
          <p className="mt-1 text-sm text-muted">
            Model confidence, not certainty.
          </p>
          <button
            type="button"
            className="mt-5 min-h-10 rounded-lg border border-line px-4 py-2 text-sm font-semibold hover:bg-slate-100"
            onClick={copyResult}
          >
            Copy result
          </button>
          {copyMessage && (
            <p className="mt-2 text-sm text-muted" role="status">
              {copyMessage}
            </p>
          )}
        </div>
      ) : (
        <p className="mt-4 rounded-lg bg-slate-100 p-4 text-sm leading-6 text-muted">
          {modelAvailable
            ? "Upload an image to see its prediction and confidence."
            : "No result is available because a compatible trained research model is not loaded."}
        </p>
      )}
    </section>
  );
}
