export interface HealthResponse {
  status: string;
  predictions_available: boolean;
  model_message: string;
}

export interface PredictionResponse {
  prediction: "glaucoma" | "normal";
  confidence: number;
  disclaimer: string;
  model_name: string;
  explanation_method: string;
  explanation_note: string;
  explanation_error: string;
  original_image: string | null;
  gradcam_heatmap: string | null;
  gradcam_overlay: string | null;
}

export interface MetricsResponse {
  available: boolean;
  message: string;
  metrics?: EvaluationMetrics | null;
}

export interface EvaluationMetrics {
  model_name: string;
  dataset_name: string;
  test_samples: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  sensitivity: number;
  specificity: number;
  roc_auc: number | null;
  confusion_matrix: [[number, number], [number, number]];
  training_history: Array<{
    epoch: number;
    train_loss: number;
    validation_loss: number;
    train_accuracy: number;
    validation_accuracy: number;
  }>;
  roc_curve: Array<{ false_positive_rate: number; true_positive_rate: number }>;
  evaluated_at: string;
}
