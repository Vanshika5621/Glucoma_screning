import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MetricsResponse } from "../types/api";

function percentage(value: number | null | undefined) {
  return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "N/A";
}

function formatEvaluatedAt(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    timeZone: "Asia/Kolkata",
    timeZoneName: "short",
  }).format(date);
}

function numberTooltip(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(2) : "N/A";
}

function percentTooltip(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${(numeric * 100).toFixed(2)}%` : "N/A";
}

export function MetricsPanel({
  response,
}: {
  response: MetricsResponse | null;
}) {
  const metrics = response?.metrics;
  const sampleCount = metrics?.test_samples ?? metrics?.evaluation_samples ?? 0;
  const positiveCount =
    metrics?.positive_samples ?? metrics?.class_counts?.glaucoma ?? 0;
  const splitCounts = metrics?.split_counts;

  return (
    <section className="panel p-6 sm:p-8" aria-labelledby="metrics-title">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="eyebrow">MODEL EVALUATION</p>
          <h2 id="metrics-title" className="mt-2 text-xl font-semibold">
            Performance dashboard
          </h2>
        </div>
        {metrics && (
          <p className="text-xs text-muted">
            {metrics.model_name} · {sampleCount} evaluated samples ·{" "}
            {positiveCount} glaucoma samples
          </p>
        )}
      </div>

      {!metrics ? (
        <p className="mt-5 rounded-lg bg-slate-100 p-4 text-sm leading-6 text-muted">
          {response?.message ??
            "Evaluation metrics are not available yet. No sample values are shown."}
        </p>
      ) : (
        <>
          <dl className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {[
              ["Recall", metrics.recall],
              ["Precision", metrics.precision],
              ["F1 score", metrics.f1_score],
              ["PR-AUC", metrics.pr_auc],
              ["ROC-AUC", metrics.roc_auc],
            ].map(([label, value]) => (
              <div className="metric-tile" key={label}>
                <dt>{label}</dt>
                <dd>{percentage(value as number | null)}</dd>
                {label === "Recall" && metrics.recall_ci_95 && (
                  <small className="mt-1 block text-muted">
                    95% CI: {percentage(metrics.recall_ci_95.lower)}–
                    {percentage(metrics.recall_ci_95.upper)}
                  </small>
                )}
              </div>
            ))}
            <div className="metric-tile">
              <dt>Accuracy</dt>
              <dd>{percentage(metrics.accuracy)}</dd>
              <small className="mt-1 block text-muted">
                Baseline (all normal) ={" "}
                {percentage(metrics.normal_baseline_accuracy)}
              </small>
            </div>
            <div className="metric-tile">
              <dt>Sensitivity</dt>
              <dd>{percentage(metrics.sensitivity)}</dd>
            </div>
            <div className="metric-tile">
              <dt>Specificity</dt>
              <dd>{percentage(metrics.specificity)}</dd>
            </div>
          </dl>

          <div className="mt-8 grid gap-8 lg:grid-cols-2">
            <div>
              <h3 className="font-semibold">Training and validation loss</h3>
              <div className="mt-3 h-64" role="img" aria-label="Loss by epoch">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics.training_history} margin={{ bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="epoch"
                      label={{ value: "Epoch", position: "insideBottom", offset: -12 }}
                    />
                    <YAxis
                      label={{ value: "Loss", angle: -90, position: "insideLeft" }}
                    />
                    <Tooltip formatter={(value) => numberTooltip(value)} />
                    <Legend />
                    <Line dataKey="train_loss" stroke="#155e63" name="Train" />
                    <Line
                      dataKey="validation_loss"
                      stroke="#a34a00"
                      name="Validation"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div>
              <h3 className="font-semibold">Training and validation accuracy</h3>
              <div
                className="mt-3 h-64"
                role="img"
                aria-label="Accuracy by epoch"
              >
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics.training_history} margin={{ bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="epoch"
                      label={{ value: "Epoch", position: "insideBottom", offset: -12 }}
                    />
                    <YAxis
                      domain={[0, 1]}
                      tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                      label={{
                        value: "Accuracy",
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip formatter={(value) => percentTooltip(value)} />
                    <Legend />
                    <Line
                      dataKey="train_accuracy"
                      stroke="#155e63"
                      name="Train"
                    />
                    <Line
                      dataKey="validation_accuracy"
                      stroke="#a34a00"
                      name="Validation"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div>
              <h3 className="font-semibold">Confusion matrix</h3>
              <p className="mt-1 text-xs text-muted">
                Rows: actual · columns: predicted · class order: normal, glaucoma
              </p>
              <div className="mt-3 grid max-w-sm grid-cols-3 gap-2 text-center text-sm">
                <span />
                <span className="font-medium">Pred. normal</span>
                <span className="font-medium">Pred. glaucoma</span>
                <span className="font-medium">Actual normal</span>
                <span className="rounded bg-teal-50 p-4">
                  {metrics.confusion_matrix[0][0]}
                </span>
                <span className="rounded bg-amber-50 p-4">
                  {metrics.confusion_matrix[0][1]}
                </span>
                <span className="font-medium">Actual glaucoma</span>
                <span className="rounded bg-amber-50 p-4">
                  {metrics.confusion_matrix[1][0]}
                </span>
                <span className="rounded bg-teal-50 p-4">
                  {metrics.confusion_matrix[1][1]}
                </span>
              </div>
            </div>
            <div>
              <h3 className="font-semibold">
                ROC curve · AUC {percentage(metrics.roc_auc)}
              </h3>
              <div className="mt-3 h-64" role="img" aria-label="ROC curve">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics.roc_curve} margin={{ bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="false_positive_rate"
                      type="number"
                      domain={[0, 1]}
                      tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                      label={{
                        value: "False Positive Rate",
                        position: "insideBottom",
                        offset: -12,
                      }}
                    />
                    <YAxis
                      dataKey="true_positive_rate"
                      type="number"
                      domain={[0, 1]}
                      tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
                      label={{
                        value: "True Positive Rate",
                        angle: -90,
                        position: "insideLeft",
                      }}
                    />
                    <Tooltip formatter={(value) => percentTooltip(value)} />
                    <ReferenceLine
                      segment={[
                        { x: 0, y: 0 },
                        { x: 1, y: 1 },
                      ]}
                      stroke="#64748b"
                      strokeDasharray="5 5"
                      ifOverflow="extendDomain"
                    />
                    <Line
                      dataKey="true_positive_rate"
                      stroke="#155e63"
                      dot={false}
                      name="ROC"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <section className="mt-8 rounded-xl border border-line p-4 sm:p-5">
            <h3 className="font-semibold">Model details</h3>
            <dl className="mt-3 grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
              <div>
                <dt className="text-muted">Model</dt>
                <dd className="font-medium">{metrics.model_name}</dd>
              </div>
              <div>
                <dt className="text-muted">Dataset</dt>
                <dd className="font-medium">{metrics.dataset_name}</dd>
              </div>
              <div>
                <dt className="text-muted">Decision threshold</dt>
                <dd className="font-medium">{percentage(metrics.threshold)}</dd>
              </div>
              {splitCounts && (
                <>
                  <div>
                    <dt className="text-muted">Train / validation / test</dt>
                    <dd className="font-medium">
                      {splitCounts.train} / {splitCounts.validation} /{" "}
                      {splitCounts.test}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-muted">Class counts</dt>
                    <dd className="font-medium">
                      {metrics.class_counts_by_split ? (
                        <>
                          {[
                            {
                              split: "Train",
                              counts: metrics.class_counts_by_split.train,
                            },
                            {
                              split: "Validation",
                              counts: metrics.class_counts_by_split.validation,
                            },
                            {
                              split: "Test",
                              counts: metrics.class_counts_by_split.test,
                            },
                          ]
                            .map(
                              ({ split, counts }) =>
                                `${split}: N ${counts.normal} / G ${counts.glaucoma}`,
                            )
                            .join(" · ")}
                        </>
                      ) : (
                        `Evaluated: Normal ${
                          metrics.class_counts?.normal ?? "N/A"
                        } / Glaucoma ${metrics.class_counts?.glaucoma ?? "N/A"}`
                      )}
                    </dd>
                  </div>
                </>
              )}
              <div className="sm:col-span-2 lg:col-span-3">
                <dt className="text-muted">Limitations</dt>
                <dd className="font-medium">
                  The held-out test set is small and imbalanced; results are
                  uncertain and are not evidence of clinical performance.
                </dd>
              </div>
            </dl>
          </section>
          <p className="mt-4 text-xs text-muted">
            Evaluated: {formatEvaluatedAt(metrics.evaluated_at)}
          </p>
        </>
      )}
    </section>
  );
}
