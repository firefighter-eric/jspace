import type { AnalysisCell, AnalysisResult, Metric } from "./types";

export const PALETTE = ["#e85f61", "#4b78ae", "#6e9f62", "#b2863f", "#8068a8", "#3d9390", "#c373a4", "#6f8293"];

export function displayToken(token: string, showWhitespace = false): string {
  const visible = token
    .replaceAll("\n", "↵")
    .replaceAll("\t", "⇥")
    .replaceAll("\r", "↵");
  if (showWhitespace) {
    return visible.replaceAll(" ", "␠") || "∅";
  }
  return visible.trim() || (token.includes(" ") ? "␠" : "∅");
}

export function normalizedToken(token: string): string {
  return token.trim().toLocaleLowerCase();
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

const RANK_COLORS = [
  "#f3cf68",
  "#9bd46c",
  "#48bf88",
  "#2a9d8f",
  "#267a85",
  "#24596d",
  "#1b3a50",
  "#111f31",
];

export function rankColor(rank: number, vocabSize: number): string {
  const normalized = Math.log(Math.max(1, rank)) / Math.log(Math.max(2, vocabSize));
  const index = Math.min(RANK_COLORS.length - 1, Math.floor(normalized * RANK_COLORS.length));
  return RANK_COLORS[index];
}

export function rankStrength(rank: number, vocabSize: number): number {
  return Math.max(
    0,
    Math.min(1, 1 - Math.log(Math.max(1, rank)) / Math.log(Math.max(2, vocabSize))),
  );
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
