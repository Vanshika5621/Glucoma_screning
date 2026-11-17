export interface PredictionHistoryItem {
  id: string;
  prediction: "glaucoma" | "normal";
  confidence: number;
  createdAt: string;
  thumbnail: string;
}

const HISTORY_STORAGE_KEY = "glaucoma-screening-history";
const MAX_HISTORY_ITEMS = 8;

function isHistoryItem(value: unknown): value is PredictionHistoryItem {
  if (!value || typeof value !== "object") return false;
  const item = value as Partial<PredictionHistoryItem>;
  return (
    typeof item.id === "string" &&
    (item.prediction === "glaucoma" || item.prediction === "normal") &&
    typeof item.confidence === "number" &&
    Number.isFinite(item.confidence) &&
    item.confidence >= 0 &&
    item.confidence <= 1 &&
    typeof item.createdAt === "string" &&
    typeof item.thumbnail === "string"
  );
}

export function loadPredictionHistory(): PredictionHistoryItem[] {
  const saved = window.localStorage.getItem(HISTORY_STORAGE_KEY);
  if (!saved) return [];

  const parsed: unknown = JSON.parse(saved);
  if (!Array.isArray(parsed) || !parsed.every(isHistoryItem)) {
    throw new Error("Saved prediction history has an invalid format.");
  }
  return parsed.slice(0, MAX_HISTORY_ITEMS);
}

export function savePredictionHistory(items: PredictionHistoryItem[]): void {
  window.localStorage.setItem(
    HISTORY_STORAGE_KEY,
    JSON.stringify(items.slice(0, MAX_HISTORY_ITEMS)),
  );
}

export function createPredictionHistoryItem(
  prediction: PredictionHistoryItem["prediction"],
  confidence: number,
  thumbnail: string,
): PredictionHistoryItem {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
    prediction,
    confidence,
    createdAt: new Date().toISOString(),
    thumbnail,
  };
}

export const MAX_SAVED_PREDICTIONS = MAX_HISTORY_ITEMS;
