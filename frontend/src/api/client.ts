import axios from "axios";
import type {
  HealthResponse,
  MetricsResponse,
  PredictionResponse,
} from "../types/api";

const client = axios.create({
  baseURL: "/api",
  timeout: 60_000,
});

export async function getHealth(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>("/health");
  return data;
}

export async function getMetrics(): Promise<MetricsResponse> {
  const { data } = await client.get<MetricsResponse>("/metrics");
  return data;
}

export async function predictImage(file: File): Promise<PredictionResponse> {
  const body = new FormData();
  body.append("file", file);
  const { data } = await client.post<PredictionResponse>("/predict", body);
  return data;
}

export async function validateImage(file: File): Promise<string> {
  const body = new FormData();
  body.append("file", file);
  const { data } = await client.post<{ message: string }>("/images/validate", body);
  return data.message;
}

export function getErrorMessage(error: unknown): string {
  if (axios.isAxiosError<{ detail?: string }>(error)) {
    return error.response?.data.detail ?? error.message;
  }
  return error instanceof Error ? error.message : "Something went wrong.";
}
