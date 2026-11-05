import type { PredictionResponse } from "../types/api";

export function ResultPanel({ result }: { result: PredictionResponse | null }) {
  const visualizationAvailable = Boolean(
    result?.original_image &&
      result.gradcam_heatmap &&
      result.gradcam_overlay,
  );

  return (
    <section className="panel p-6 sm:p-8" aria-labelledby="result-title">
      <p className="eyebrow">LATEST RESULT</p>
      <h2 id="result-title" className="mt-2 text-xl font-semibold">
        Research prediction
      </h2>
      {result ? (
        <div className="mt-5">
          <p className="text-2xl font-bold capitalize">{result.prediction}</p>
          <p className="mt-2 text-sm text-muted">
            Model score: {(result.confidence * 100).toFixed(1)}%
          </p>
          <p className="mt-1 text-xs text-muted">Model: {result.model_name}</p>
          <div className="mt-6">
            <h3 className="text-base font-semibold">Model explanation</h3>
            <p className="mt-1 text-xs text-muted">{result.explanation_method}</p>
            {!visualizationAvailable ? (
              <p
                className="mt-3 rounded-lg bg-amber-50 p-3 text-sm text-amber-900"
                role="status"
              >
                {result.explanation_error ||
                  "Grad-CAM visualization is unavailable for this result."}
              </p>
            ) : (
              <>
                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  {[
                    ["Original image", result.original_image],
                    ["Grad-CAM heatmap", result.gradcam_heatmap],
                    ["Overlay", result.gradcam_overlay],
                  ].map(([label, image]) => (
                    <figure key={label} className="min-w-0">
                      {image ? (
                        <img
                          className="aspect-square w-full rounded-lg border border-line bg-slate-50 object-contain"
                          src={image}
                          alt={`${label} for the uploaded retinal image`}
                        />
                      ) : (
                        <div
                          className="flex aspect-square items-center justify-center rounded-lg border border-line bg-slate-50 p-3 text-center text-sm text-muted"
                          role="status"
                        >
                          {label} unavailable
                        </div>
                      )}
                      <figcaption className="mt-2 text-center text-xs font-medium">
                        {label}
                      </figcaption>
                    </figure>
                  ))}
                </div>
                <div
                  className="mt-4 rounded-lg border border-line bg-white p-3"
                  aria-label="Grad-CAM heatmap legend"
                >
                  <p className="text-xs font-semibold">
                    Relative Grad-CAM activation
                  </p>
                  <div
                    className="mt-2 h-3 rounded"
                    aria-hidden="true"
                    style={{
                      background:
                        "linear-gradient(to right, #313695, #74add1, #ffffbf, #f46d43, #a50026)",
                    }}
                  />
                  <div className="mt-1 flex justify-between text-xs text-muted">
                    <span>Lower relative response</span>
                    <span>Higher relative response</span>
                  </div>
                </div>
              </>
            )}
            <p className="mt-3 text-xs leading-5 text-muted">
              {result.explanation_note}
            </p>
          </div>
          <p className="mt-4 rounded-lg bg-amber-50 p-4 text-sm leading-6 text-amber-900">
            {result.disclaimer}
          </p>
        </div>
      ) : (
        <p className="mt-4 rounded-lg bg-slate-100 p-4 text-sm leading-6 text-muted">
          No prediction is available yet. Results will appear here only after a
          compatible research model is trained and configured.
        </p>
      )}
    </section>
  );
}
