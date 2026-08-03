import type { AnalysisResult, RuntimeStatus } from "./types";

async function readJson<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as T & { error?: string };
  if (!response.ok) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }
  return payload;
}

export async function fetchRuntimeStatus(signal?: AbortSignal): Promise<RuntimeStatus> {
  const response = await fetch("/api/status", { signal, cache: "no-store" });
  return readJson<RuntimeStatus>(response);
}

export async function analyzePrompt(prompt: string): Promise<AnalysisResult> {
  const response = await fetch("/api/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, top_k: 10, max_tokens: 64 }),
  });
  return readJson<AnalysisResult>(response);
}
