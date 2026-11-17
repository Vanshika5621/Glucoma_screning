import type { PredictionHistoryItem } from "../history";

export function HistoryPanel({
  items,
  onClear,
}: {
  items: PredictionHistoryItem[];
  onClear: () => void;
}) {
  return (
    <section className="panel mt-6 p-6 sm:p-8" aria-labelledby="history-title">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 id="history-title" className="text-xl font-semibold">
            Prediction history
          </h2>
          <p className="mt-1 text-sm text-muted">
            Saved only in this browser on this device.
          </p>
        </div>
        {items.length > 0 && (
          <button
            type="button"
            className="min-h-10 rounded-lg border border-line px-4 py-2 text-sm font-semibold hover:bg-slate-100"
            onClick={onClear}
          >
            Clear history
          </button>
        )}
      </div>

      {items.length === 0 ? (
        <p className="mt-5 text-sm text-muted">No predictions saved yet.</p>
      ) : (
        <ul className="mt-5 divide-y divide-line">
          {items.map((item) => (
            <li
              className="flex items-center gap-4 py-4 first:pt-0 last:pb-0"
              key={item.id}
            >
              {item.thumbnail ? (
                <img
                  className="h-14 w-14 rounded-lg border border-line object-cover"
                  src={item.thumbnail}
                  alt=""
                />
              ) : (
                <div
                  className="flex h-14 w-14 items-center justify-center rounded-lg bg-slate-100 text-xs text-muted"
                  role="img"
                  aria-label="Thumbnail unavailable"
                >
                  Image
                </div>
              )}
              <div className="min-w-0 flex-1">
                <p className="font-semibold">
                  {item.prediction === "glaucoma" ? "Yes" : "No"}
                  <span className="ml-2 text-sm font-normal text-muted">
                    Confidence {(item.confidence * 100).toFixed(1)}%
                  </span>
                </p>
                <time
                  className="mt-1 block text-sm text-muted"
                  dateTime={item.createdAt}
                >
                  {new Intl.DateTimeFormat(undefined, {
                    dateStyle: "medium",
                    timeStyle: "short",
                  }).format(new Date(item.createdAt))}
                </time>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
