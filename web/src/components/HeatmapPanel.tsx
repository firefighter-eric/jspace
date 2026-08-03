import { Gauge, Sparkles } from "lucide-react";
import { displayToken, heatColor } from "../analysis";
import type { AnalysisResult, Metric } from "../types";

type Props = {
  result: AnalysisResult | null;
  metric: Metric;
  selectedLayer: number;
  selectedPosition: number;
  pinned: Set<number>;
  runState: "idle" | "running" | "success" | "error";
  onMetricChange: (metric: Metric) => void;
  onSelect: (layer: number, position: number, tokenId?: number) => void;
};

export function HeatmapPanel({
  result,
  metric,
  selectedLayer,
  selectedPosition,
  pinned,
  runState,
  onMetricChange,
  onSelect,
}: Props) {
  return (
    <section className="heatmap-panel">
      <div className="analysis-heading">
        <div><span className="eyebrow">PRIMARY VIEW</span><h1>Layer × Position</h1></div>
        <div className="metric-switch" aria-label="指标切换">
          <button type="button" className={metric === "rank" ? "active" : ""} onClick={() => onMetricChange("rank")}><Gauge size={14} />Full-vocab rank</button>
          <button type="button" className={metric === "probability" ? "active" : ""} onClick={() => onMetricChange("probability")}><Sparkles size={14} />Probability</button>
        </div>
      </div>

      {!result ? (
        <div className={`analysis-empty ${runState === "running" ? "loading" : ""}`}>
          <div className="empty-matrix" aria-hidden="true">
            {Array.from({ length: 54 }, (_, index) => <i key={index} />)}
          </div>
          <strong>{runState === "running" ? "正在计算真实残差读出" : "尚无模型分析结果"}</strong>
          <span>{runState === "running" ? "Qwen 前向计算 → Jacobian 传输 → 全词表排名" : "输入提示词后点击“开始真实探查”"}</span>
        </div>
      ) : (
        <>
          <div className="legend-row">
            <span>{metric === "rank" ? "可读 Top-1 的完整词表排名" : "可读 Top-1 的完整词表概率"}</span>
            <div className="legend-scale" />
            <span>低</span><span>高</span>
          </div>
          <div className="heatmap-scroll">
            <div className="token-axis" style={{ gridTemplateColumns: `44px repeat(${result.tokens.length}, minmax(28px, 1fr))` }}>
              <span />
              {result.tokens.map((token) => (
                <button
                  key={`${token.id}-${token.index}`}
                  type="button"
                  className={selectedPosition === token.index ? "active" : ""}
                  title={`${token.index}: ${displayToken(token.text)} · token ${token.id}`}
                  onClick={() => onSelect(selectedLayer, token.index)}
                >
                  <small>{token.index}</small><span>{displayToken(token.text)}</span>
                </button>
              ))}
            </div>
            <div className="heatmap" style={{ gridTemplateColumns: `44px repeat(${result.tokens.length}, minmax(28px, 1fr))` }}>
              {result.layers.flatMap((layer, rowIndex) => [
                <button
                  key={`label-${layer}`}
                  type="button"
                  className={`layer-label ${selectedLayer === layer ? "active" : ""}`}
                  onClick={() => onSelect(layer, selectedPosition)}
                >{layer}</button>,
                ...result.cells[rowIndex].map((cell, position) => (
                  <button
                    key={`${layer}-${position}`}
                    type="button"
                    aria-label={`Layer ${layer}, Position ${position}, ${displayToken(cell.top_token)}, full-vocab rank ${cell.top_rank}`}
                    title={`L${layer} · P${position} · ${displayToken(cell.top_token)} · full-vocab #${cell.top_rank} · ${(cell.top_probability * 100).toFixed(3)}%`}
                    className={`heat-cell ${cell.changed ? "changed" : ""} ${pinned.has(cell.top_id) ? "pinned" : ""} ${selectedLayer === layer && selectedPosition === position ? "selected" : ""}`}
                    style={{ backgroundColor: heatColor(cell, metric, result.vocab_size) }}
                    onClick={() => onSelect(layer, position, cell.top_id)}
                  >
                    <span>{displayToken(cell.top_token)}</span>
                    {cell.top_rank > 1 ? <sup>{cell.top_rank}</sup> : null}
                  </button>
                )),
              ])}
            </div>
          </div>
        </>
      )}
    </section>
  );
}
