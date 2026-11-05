import { Suspense, lazy, useEffect, useState } from "react";
import { getErrorMessage, getHealth, getMetrics } from "./api/client";
import { ResultPanel } from "./components/ResultPanel";
import { UploadPanel } from "./components/UploadPanel";
import type {
  HealthResponse,
  MetricsResponse,
  PredictionResponse,
} from "./types/api";

const MetricsPanel = lazy(() => import("./components/MetricsPanel").then(
  (module) => ({ default: module.MetricsPanel }),
));

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch((error: unknown) => setHealthError(getErrorMessage(error)));
    getMetrics()
      .then(setMetrics)
      .catch(() =>
        setMetrics({
          available: false,
          message: "Evaluation metrics could not be loaded. Refresh to try again.",
        }),
      );
  }, []);

  return (
    <main className="mx-auto min-h-screen max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <header className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="eyebrow">LOCAL RESEARCH PROTOTYPE</p>
          <h1 className="mt-2 text-3xl font-bold tracking-tight">
            Glaucoma image research dashboard
          </h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-muted">
            Experimental classification of retinal fundus photographs. This
            dashboard is not a medical device and cannot diagnose glaucoma.
          </p>
        </div>
        <div
          className="rounded-full border border-line bg-white px-4 py-2 text-sm"
          role="status"
          aria-live="polite"
        >
          <span
            className={`mr-2 inline-block h-2.5 w-2.5 rounded-full ${
              health?.status === "ok" ? "bg-emerald-600" : "bg-slate-400"
            }`}
          />
          {health?.status === "ok" ? "API online" : "Checking API…"}
        </div>
      </header>

      {(healthError || health?.model_message) && (
        <div
          className="mb-6 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm leading-6 text-amber-950"
          role="status"
        >
          <strong>Model status:</strong>{" "}
          {healthError ?? health?.model_message}
          {!health?.predictions_available &&
            " No prediction or confidence value will be shown until a model is configured."}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-2">
        <UploadPanel
          predictionsAvailable={health?.predictions_available ?? false}
          onPrediction={setPrediction}
        />
        <ResultPanel result={prediction} />
      </div>

      {metrics?.available ? (
        <div className="mt-6">
          <Suspense
            fallback={
              <p className="panel p-6 text-sm text-muted" role="status">
                Loading evaluation charts…
              </p>
            }
          >
            <MetricsPanel response={metrics} />
          </Suspense>
        </div>
      ) : (
        <section
          className="panel mt-6 p-6 sm:p-8"
          aria-labelledby="metrics-title"
        >
          <p className="eyebrow">MODEL EVALUATION</p>
          <h2 id="metrics-title" className="mt-2 text-xl font-semibold">
            Performance dashboard
          </h2>
          <p className="mt-5 rounded-lg bg-slate-100 p-4 text-sm leading-6 text-muted">
            {metrics?.message ?? "Loading evaluation report…"}
          </p>
        </section>
      )}

      <footer className="mt-8 flex flex-wrap justify-between gap-3 border-t border-line pt-5 text-sm text-muted">
        <p>
          For learning and research only. Never use results for diagnosis or
          treatment decisions.
        </p>
        <a
          className="font-semibold text-primary underline underline-offset-4"
          href="/docs"
        >
          API documentation
        </a>
      </footer>
    </main>
  );
}
