import { Download, Pin } from "lucide-react";
import { displayToken, formatProbability, selectedCell } from "../analysis";
import type { AnalysisResult, Metric } from "../types";

type Props = {
  result: AnalysisResult | null;
  metric: Metric;
  selectedLayer: number;
  selectedPosition: number;
  pinned: Set<number>;
  onTogglePin: (tokenId: number) => void;
  onExport: () => void;
};

export function InspectorPanel({
  result,
  metric,
  selectedLayer,
  selectedPosition,
  pinned,
  onTogglePin,
  onExport,
}: Props) {
  const cell = result ? selectedCell(result, selectedLayer, selectedPosition) : null;
  const inputToken = result?.tokens[Math.min(selectedPosition, result.tokens.length - 1)];
  const topProbability = cell?.candidates[0]?.probability ?? 1;
  return (
    <aside className="inspector">
      <div className="inspector-heading">
        <div><span className="eyebrow">SELECTION</span><h2>Token 检查器</h2></div>
        <button className="icon-button small" type="button" onClick={onExport} aria-label="导出当前单元" disabled={!result}><Download size={16} /></button>
      </div>
      {!result || !cell ? (
        <div className="inspector-empty">
          <strong>等待真实候选</strong>
          <p>自定义输入不会继承官方样本词。完成模型分析后，这里才会显示该单元按官方规则筛选的 Top-10。</p>
        </div>
      ) : (
        <>
          <div className="selection-readout"><span>Layer <strong>{selectedLayer}</strong></span><i /> <span>Position <strong>{selectedPosition}</strong></span></div>
          <div className="selected-token"><span>输入 token</span><strong title={inputToken ? `token ${inputToken.id}` : undefined}>{inputToken ? displayToken(inputToken.text) : "—"}</strong></div>
          <section className="candidates">
            <div className="subheading"><span>可读候选 token</span><small>Full-vocab rank · Probability</small></div>
            {cell.candidates.map((item, index) => {
              const canTrack = result.tracked_token_ids.includes(item.id);
              return (
              <div className="candidate" key={item.id}>
                <span className="candidate-rank">{String(index + 1).padStart(2, "0")}</span>
                <button type="button" className="candidate-token" title={`token ${item.id}`} onClick={() => onTogglePin(item.id)} disabled={!canTrack}>{displayToken(item.token)}</button>
                <div className="strength"><i style={{ width: `${Math.max(2, (item.probability / topProbability) * 100)}%` }} /></div>
                <span className="rank-value">{metric === "probability" ? formatProbability(item.probability) : `#${item.rank}`}</span>
                <button type="button" className={`pin-button ${pinned.has(item.id) ? "active" : ""}`} aria-label={`${pinned.has(item.id) ? "取消固定" : "固定"} ${displayToken(item.token)}`} title={canTrack ? "查看完整词表 rank 轨迹" : "该 token 未进入本次 rank 追踪集合"} onClick={() => onTogglePin(item.id)} disabled={!canTrack}><Pin size={14} fill={pinned.has(item.id) ? "currentColor" : "none"} /></button>
              </div>
              );
            })}
          </section>
          <section className="explanation">
            <div className="subheading"><span>如何阅读</span><small>真实 J-lens</small></div>
            <p>第 <strong>{selectedLayer}</strong> 层、位置 <strong>{selectedPosition}</strong> 的残差已通过拟合 Jacobian 传输到最终层空间，再由 Qwen 的 unembed 对完整词表排名。</p>
            <p>候选显示遵循 Anthropic 的 <code>mask_display=True</code>：只隐藏标点、空白和特殊 token，右侧 Rank 与 Probability 仍基于未过滤的完整词表。</p>
          </section>
          <section className="provenance">
            <div><span>模型</span><strong>{result.model}</strong></div>
            <div><span>透镜</span><strong>{result.lens_prompts} prompts</strong></div>
            <div><span>计算设备</span><strong>{result.device.toUpperCase()}</strong></div>
            <div><span>数据模式</span><strong className="real">真实读数</strong></div>
          </section>
        </>
      )}
    </aside>
  );
}
