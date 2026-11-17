import { useEffect, useState } from "react";
import { getHealth } from "./api/client";
import { HistoryPanel } from "./components/HistoryPanel";
import { ResultPanel } from "./components/ResultPanel";
import { UploadPanel } from "./components/UploadPanel";
import {
  createPredictionHistoryItem,
  loadPredictionHistory,
  MAX_SAVED_PREDICTIONS,
  savePredictionHistory,
} from "./history";
import type { PredictionHistoryItem } from "./history";
import { createThumbnail } from "./thumbnail";
import type {
  HealthResponse,
  PredictionResponse,
} from "./types/api";

const THEME_STORAGE_KEY = "glaucoma-screening-theme";

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [initialHistory] = useState(() => {
    try {
      return { items: loadPredictionHistory(), error: null };
    } catch {
      return {
        items: [],
        error: "Saved history could not be loaded from this browser.",
      };
    }
  });
  const [history, setHistory] = useState<PredictionHistoryItem[]>(
    initialHistory.items,
  );
  const [historyMessage, setHistoryMessage] = useState<string | null>(
    initialHistory.error,
  );
  const [isDark, setIsDark] = useState(() => {
    try {
      return window.localStorage.getItem(THEME_STORAGE_KEY) === "dark";
    } catch {
      return false;
    }
  });

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = isDark ? "dark" : "light";
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, isDark ? "dark" : "light");
    } catch {
      setHistoryMessage("Theme preference could not be saved in this browser.");
    }
  }, [isDark]);

  useEffect(() => {
    try {
      savePredictionHistory(history);
    } catch {
      setHistoryMessage(
        "History could not be saved. Browser storage may be full or unavailable.",
      );
    }
  }, [history]);

  async function handlePrediction(
    result: PredictionResponse | null,
    file?: File,
  ) {
    setPrediction(result);
    if (!result || !file) return;

    setHistoryMessage(null);
    let thumbnail = "";
    try {
      thumbnail = await createThumbnail(file);
    } catch {
      setHistoryMessage("Prediction saved without its image thumbnail.");
    }
    const item = createPredictionHistoryItem(
      result.prediction,
      result.confidence,
      thumbnail,
    );
    setHistory((previous) =>
      [item, ...previous].slice(0, MAX_SAVED_PREDICTIONS),
    );
  }

  function clearHistory() {
    setHistory([]);
    setHistoryMessage(null);
  }

  return (
    <main className="mx-auto flex min-h-screen max-w-5xl flex-col px-4 py-8 sm:px-6 lg:px-8">
      <header className="mb-8 flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-3xl font-bold tracking-tight">Glaucoma screening</h1>
        <button
          type="button"
          className="min-h-10 rounded-lg border border-line px-4 py-2 text-sm font-semibold hover:bg-slate-100"
          aria-pressed={isDark}
          onClick={() => setIsDark((current) => !current)}
        >
          {isDark ? "Light mode" : "Dark mode"}
        </button>
      </header>

      <div className="grid gap-6 lg:grid-cols-2">
        <UploadPanel
          predictionsAvailable={health?.predictions_available ?? false}
          onPrediction={handlePrediction}
        />
        <ResultPanel
          result={prediction}
          modelAvailable={health?.predictions_available ?? false}
        />
      </div>

      <HistoryPanel items={history} onClear={clearHistory} />
      {historyMessage && (
        <p className="mt-3 text-sm text-muted" role="status">
          {historyMessage}
        </p>
      )}

      <p className="mt-6 text-center text-sm text-muted">
        Research use only. Not for diagnosis or treatment.
      </p>
    </main>
  );
}
