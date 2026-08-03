import { Braces, CircleHelp, Eye } from "lucide-react";
import { displayToken, PALETTE, rankTrajectory } from "../analysis";
import type { AnalysisResult } from "../types";

type Props = {
  result: AnalysisResult | null;
  selectedLayer: number;
  selectedPosition: number;
  pinned: Set<number>;
};

function pathForRanks(values: number[], vocabSize: number, width: number, height: number) {
  const logCeiling = Math.log(Math.max(2, vocabSize));
  return values.map((value, index) => {
    const x = values.length === 1 ? 0 : (index / (values.length - 1)) * width;
    const y = (Math.log(Math.max(1, value)) / logCeiling) * (height - 10);
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

export function TracePanels({ result, selectedLayer, selectedPosition, pinned }: Props) {
  if (!result) {
    return (
      <section className="lower-grid empty">
        <div className="trajectory-panel"><div className="panel-title"><div><span className="eyebrow">TRACE</span><h2>固定 Token 排名轨迹</h2></div><Eye size={17} /></div><p className="panel-empty-copy">运行后点击热力图单元或图钉，查看完整词表 rank 轨迹。</p></div>
        <div className="output-panel"><div className="panel-title"><div><span className="eyebrow">FINAL LAYER</span><h2>真实下一 Token</h2></div><Braces size={17} /></div><p className="panel-empty-copy">运行后显示最终层未经筛选的下一 token 分布。</p></div>
      </section>
    );
  }

  const tracked = [...pinned]
    .filter((tokenId) => result.rank_tracks[String(tokenId)])
    .slice(0, PALETTE.length)
    .map((tokenId) => ({
      id: tokenId,
      token: result.vocab[String(tokenId)] ?? `[${tokenId}]`,
      ranks: rankTrajectory(result, selectedPosition, tokenId),
    }));
  const currentLayerIndex = Math.max(0, result.layers.indexOf(selectedLayer));
  const finalOutput = result.final_outputs[selectedPosition];
  return (
    <section className="lower-grid">
      <div className="trajectory-panel">
        <div className="panel-title"><div><span className="eyebrow">TRACE</span><h2>固定 Token 排名轨迹</h2></div><Eye size={17} /></div>
        {tracked.length ? <><div className="chart-wrap">
          <div className="chart-y"><span>#1</span><span>10³</span><span>词表底部</span></div>
          <svg viewBox="0 0 420 150" role="img" aria-label="真实候选概念随层变化折线图">
            <g className="grid-lines"><line x1="0" x2="420" y1="20" y2="20"/><line x1="0" x2="420" y1="75" y2="75"/><line x1="0" x2="420" y1="130" y2="130"/></g>
            <line className="current-line" x1={(currentLayerIndex / Math.max(1, result.layers.length - 1)) * 420} x2={(currentLayerIndex / Math.max(1, result.layers.length - 1)) * 420} y1="0" y2="150" />
            {tracked.map((item, index) => <path key={item.id} className="real-trace" style={{ stroke: PALETTE[index] }} d={pathForRanks(item.ranks, result.vocab_size, 420, 130)} />)}
          </svg>
        </div>
        <div className="chart-footer"><span>Layer {result.layers[0]}</span><div className="line-key">{tracked.map((item, index) => <span key={item.id}><i style={{ background: PALETTE[index] }} />{displayToken(item.token)}</span>)}</div><span>Layer {result.layers.at(-1)}</span></div>
        </> : <p className="panel-empty-copy">点击热力图单元或候选右侧图钉，固定最多 {PALETTE.length} 个可追踪 token。</p>}
      </div>
      <div className="output-panel">
        <div className="panel-title"><div><span className="eyebrow">FINAL LAYER</span><h2>真实下一 Token</h2></div><Braces size={17} /></div>
        <p className="output-answer">未经筛选 Top-1：<strong>{displayToken(finalOutput.top_token)}</strong></p>
        <div className="output-tokens">
          {finalOutput.candidates.slice(0, 4).map((candidate, index) => <span key={candidate.id} className={index === 0 ? "intermediate" : ""} title={`${(candidate.probability * 100).toFixed(3)}%`}>{displayToken(candidate.token)}</span>)}
        </div>
        <div className="output-note"><CircleHelp size={15} /><span>来自 Layer {result.layers.at(-1)} 的原始完整词表分布，不使用 word-like 显示过滤。</span></div>
      </div>
    </section>
  );
}
