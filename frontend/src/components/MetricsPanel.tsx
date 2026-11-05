import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { MetricsResponse } from "../types/api";

function percentage(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function MetricsPanel({
  response,
}: {
  response: MetricsResponse | null;
}) {
  const metrics = response?.metrics;

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
            {metrics.model_name} · {metrics.test_samples} held-out samples
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
              ["Accuracy", metrics.accuracy],
              ["Precision", metrics.precision],
              ["Recall", metrics.recall],
              ["F1 score", metrics.f1_score],
              ["Sensitivity", metrics.sensitivity],
              ["Specificity", metrics.specificity],
              ["ROC-AUC", metrics.roc_auc],
            ].map(([label, value]) => (
              <div className="metric-tile" key={label}>
                <dt>{label}</dt>
                <dd>
                  {typeof value === "number" ? percentage(value) : "N/A"}
                </dd>
              </div>
            ))}
          </dl>

          <div className="mt-8 grid gap-8 lg:grid-cols-2">
            <div>
              <h3 className="font-semibold">Training and validation loss</h3>
              <div className="mt-3 h-64" role="img" aria-label="Loss by epoch">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics.training_history}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis />
                    <Tooltip />
                    <Line dataKey="train_loss" stroke="#155e63" name="Train loss" />
                    <Line
                      dataKey="validation_loss"
                      stroke="#a34a00"
                      name="Validation loss"
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
                  <LineChart data={metrics.training_history}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="epoch" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Line
                      dataKey="train_accuracy"
                      stroke="#155e63"
                      name="Train accuracy"
                    />
                    <Line
                      dataKey="validation_accuracy"
                      stroke="#a34a00"
                      name="Validation accuracy"
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
          </div>

          <div className="mt-8">
            <h3 className="font-semibold">ROC curve</h3>
            <div className="mt-3 h-64" role="img" aria-label="ROC curve">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={metrics.roc_curve}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="false_positive_rate"
                    type="number"
                    domain={[0, 1]}
                  />
                  <YAxis
                    dataKey="true_positive_rate"
                    type="number"
                    domain={[0, 1]}
                  />
                  <Tooltip />
                  <Line
                    dataKey="true_positive_rate"
                    stroke="#155e63"
                    dot={false}
                    name="True positive rate"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <p className="mt-4 text-xs text-muted">
            Dataset: {metrics.dataset_name} · Evaluated: {metrics.evaluated_at}
          </p>
        </>
      )}
    </section>
  );
}
