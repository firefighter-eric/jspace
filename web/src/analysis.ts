import type { AnalysisCell, AnalysisResult, Metric } from "./types";

export const PALETTE = ["#42ced7", "#ff8069", "#7c91ff", "#d9b65d", "#7bd88f", "#d48cff", "#f19d5c", "#72a7ff"];

export function displayToken(token: string): string {
  if (token === " ") return "␠";
  if (token === "\n") return "↵";
  if (token === "\n\n") return "↵↵";
  return token.replaceAll("\n", "↵").replaceAll("\t", "⇥").trim() || "∅";
}

export function layerIndex(result: AnalysisResult, layer: number): number {
  return Math.max(0, result.layers.indexOf(layer));
}

export function selectedCell(
  result: AnalysisResult,
  layer: number,
  position: number,
): AnalysisCell {
  const rowIndex = layerIndex(result, layer);
  const safePosition = Math.max(0, Math.min(position, result.tokens.length - 1));
  return result.cells[rowIndex][safePosition];
}

export function metricValue(cell: AnalysisCell, metric: Metric, vocabSize: number): number {
  if (metric === "probability") {
    return Math.max(0, Math.min(1, Math.sqrt(cell.top_probability) * 1.5));
  }
  const ceiling = Math.max(2, vocabSize);
  return Math.max(0, Math.min(1, 1 - Math.log(Math.max(1, cell.top_rank)) / Math.log(ceiling)));
}

export function heatColor(cell: AnalysisCell, metric: Metric, vocabSize: number): string {
  const value = metricValue(cell, metric, vocabSize);
  return `rgba(44, 203, 214, ${0.04 + value * 0.65})`;
}

export function rankTrajectory(
  result: AnalysisResult,
  position: number,
  tokenId: number,
): number[] {
  const safePosition = Math.max(0, Math.min(position, result.tokens.length - 1));
  const track = result.rank_tracks[String(tokenId)];
  return track ? track.map((layer) => layer[safePosition]) : [];
}

export function formatProbability(value: number): string {
  if (value >= 0.01) return `${(value * 100).toFixed(2)}%`;
  if (value >= 0.0001) return `${(value * 100).toFixed(3)}%`;
  return value.toExponential(2);
}
