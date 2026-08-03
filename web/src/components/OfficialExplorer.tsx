import {
  ChartNoAxesCombined,
  CircleHelp,
  Grid3X3,
  Keyboard,
  Pin,
  ScanSearch,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, type CSSProperties } from "react";
import {
  displayToken,
  layerIndex,
  PALETTE,
  rankColor,
  rankTrajectory,
} from "../analysis";
import type { AnalysisResult, Candidate, RunState } from "../types";

type ExplorerProps = {
  result: AnalysisResult | null;
  runState: RunState;
  selectedLayer: number;
  selectedPosition: number;
  pinned: Set<number>;
  activePinnedToken: number | null;
  showWhitespace: boolean;
  onSelect: (layer: number, position: number) => void;
  onPin: (tokenId: number) => void;
  onActivatePin: (tokenId: number) => void;
  onClearPins: () => void;
  onShowWhitespaceChange: (value: boolean) => void;
};

type SliceTableProps = {
  title: string;
  subtitle: string;
  rows: Array<{
    key: string;
    label: string;
    context?: string;
    selected: boolean;
    candidates: Candidate[];
    onSelect: () => void;
  }>;
  pinned: Set<number>;
  showWhitespace: boolean;
  onPin: (tokenId: number) => void;
};

function tokenLabel(token: string, showWhitespace: boolean) {
  return displayToken(token, showWhitespace);
}

function compactRank(rank: number) {
  if (rank < 1_000) return String(rank);
  if (rank < 1_000_000) return `${(rank / 1_000).toFixed(rank < 10_000 ? 1 : 0)}k`;
  return `${(rank / 1_000_000).toFixed(1)}m`;
}

function SliceTable({
  title,
  subtitle,
  rows,
  pinned,
  showWhitespace,
  onPin,
}: SliceTableProps) {
  return (
    <section className="slice-table-card">
      <header className="card-heading compact-heading">
        <div><span className="eyebrow">TOP-10 SLICE</span><h2>{title}</h2></div>
        <span className="card-meta">{subtitle}</span>
      </header>
      <div className="slice-table-scroll">
        {rows.map((row) => (
          <div
            key={row.key}
            className={`slice-row ${row.selected ? "selected" : ""}`}
            role="button"
            tabIndex={0}
            onClick={row.onSelect}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") row.onSelect();
            }}
          >
            <span className="slice-row-label">{row.label}</span>
            {row.context ? <span className="slice-row-context" title={row.context}>{row.context}</span> : null}
            <div className="slice-candidates">
              {row.candidates.map((candidate) => (
                <button
                  type="button"
                  key={candidate.id}
                  className={pinned.has(candidate.id) ? "pinned" : ""}
                  title={`token ${candidate.id} · full-vocab rank #${candidate.rank}`}
                  onClick={(event) => {
                    event.stopPropagation();
                    onPin(candidate.id);
                  }}
                >
                  <span>{tokenLabel(candidate.token, showWhitespace)}</span>
                  {candidate.rank > 1 ? <sup>{compactRank(candidate.rank)}</sup> : null}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function TokenMatrix({
  result,
  selectedLayer,
  selectedPosition,
  pinned,
  showWhitespace,
  onSelect,
  onPin,
}: Omit<ExplorerProps, "runState" | "activePinnedToken" | "onActivatePin" | "onClearPins" | "onShowWhitespaceChange"> & { result: AnalysisResult }) {
  const reversedLayers = useMemo(() => [...result.layers].reverse(), [result.layers]);
  const columns = `38px repeat(${result.tokens.length}, minmax(46px, 1fr))`;
  const minWidth = Math.max(660, 38 + result.tokens.length * 48);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = scrollRef.current;
    if (!element || result.tokens.length <= 1) return;
    const progress = selectedPosition / (result.tokens.length - 1);
    element.scrollLeft = progress * Math.max(0, element.scrollWidth - element.clientWidth);
  }, [result.tokens.length, selectedPosition]);

  return (
    <section className="matrix-card">
      <header className="card-heading">
        <div><span className="eyebrow">PRIMARY VIEW</span><h1>Layer × Position token grid</h1></div>
        <div className="matrix-legend" aria-label="排名图例">
          <span>Rank 1</span>
          {[1, 5, 25, 125, 625, 3_125, 15_625, 78_125].map((rank) => (
            <i key={rank} style={{ background: rankColor(rank, result.vocab_size) }} />
          ))}
          <span>词表底部</span>
        </div>
      </header>
      <div className="matrix-scroll" ref={scrollRef}>
        <div className="matrix-grid" style={{ gridTemplateColumns: columns, minWidth }}>
          {reversedLayers.flatMap((layer) => {
            const rowIndex = layerIndex(result, layer);
            return [
              <button
                type="button"
                key={`layer-${layer}`}
                className={`matrix-layer ${selectedLayer === layer ? "selected" : ""}`}
                onClick={() => onSelect(layer, selectedPosition)}
                aria-label={`选择第 ${layer} 层`}
              >
                {layer}
              </button>,
              ...result.cells[rowIndex].map((cell, position) => (
                <button
                  type="button"
                  key={`${layer}-${position}`}
                  className={`matrix-token ${selectedLayer === layer && selectedPosition === position ? "selected" : ""} ${pinned.has(cell.top_id) ? "pinned" : ""}`}
                  style={{ "--rank-color": rankColor(cell.top_rank, result.vocab_size) } as CSSProperties}
                  title={`L${layer} · P${position} · token ${cell.top_id} · full-vocab #${cell.top_rank}`}
                  aria-label={`Layer ${layer}, Position ${position}, ${tokenLabel(cell.top_token, true)}, rank ${cell.top_rank}`}
                  onClick={() => onSelect(layer, position)}
                  onDoubleClick={() => onPin(cell.top_id)}
                  onMouseEnter={(event) => {
                    if (event.shiftKey) onSelect(layer, position);
                  }}
                >
                  <span>{tokenLabel(cell.top_token, showWhitespace)}</span>
                  {cell.top_rank > 1 ? <sup>{compactRank(cell.top_rank)}</sup> : null}
                </button>
              )),
            ];
          })}
        </div>
        <div className="matrix-axis" style={{ gridTemplateColumns: columns, minWidth }}>
          <span />
          {result.tokens.map((token) => (
            <button
              type="button"
              key={`${token.id}-${token.index}`}
              className={selectedPosition === token.index ? "selected" : ""}
              onClick={() => onSelect(selectedLayer, token.index)}
              title={`Position ${token.index} · token ${token.id}`}
            >
              <span>{tokenLabel(token.text, showWhitespace)}</span><small>{token.index}</small>
            </button>
          ))}
        </div>
      </div>
      <p className="matrix-hint">单击选择单元 · 双击固定该单元的 Top‑1 token · 红色上标为完整词表排名</p>
    </section>
  );
}

function PinnedTokens({
  result,
  pinned,
  activePinnedToken,
  showWhitespace,
  onPin,
  onActivatePin,
  onClearPins,
}: Pick<ExplorerProps, "pinned" | "activePinnedToken" | "showWhitespace" | "onPin" | "onActivatePin" | "onClearPins"> & { result: AnalysisResult }) {
  const ids = [...pinned].filter((tokenId) => result.rank_tracks[String(tokenId)]);
  return (
    <div className="pinned-toolbar">
      <span><Pin size={13} />固定 token</span>
      <div className="pin-list">
        {ids.map((tokenId, index) => (
          <button
            type="button"
            key={tokenId}
            className={activePinnedToken === tokenId ? "active" : ""}
            style={{ "--pin-color": PALETTE[index % PALETTE.length] } as CSSProperties}
            onClick={() => onActivatePin(tokenId)}
            title={`token ${tokenId}`}
          >
            <i />{tokenLabel(result.vocab[String(tokenId)] ?? `[${tokenId}]`, showWhitespace)}
            <span
              role="button"
              tabIndex={0}
              aria-label="取消固定"
              onClick={(event) => {
                event.stopPropagation();
                onPin(tokenId);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.stopPropagation();
                  onPin(tokenId);
                }
              }}
            ><X size={11} /></span>
          </button>
        ))}
        {ids.length ? <button type="button" className="clear-pins" onClick={onClearPins}>全部清除</button> : <small>点击右侧候选即可固定</small>}
      </div>
    </div>
  );
}

function RankHeatmap({
  result,
  tokenId,
  selectedLayer,
  selectedPosition,
  showWhitespace,
  onSelect,
}: {
  result: AnalysisResult;
  tokenId: number | null;
  selectedLayer: number;
  selectedPosition: number;
  showWhitespace: boolean;
  onSelect: (layer: number, position: number) => void;
}) {
  const track = tokenId == null ? null : result.rank_tracks[String(tokenId)];
  const reversedLayers = useMemo(() => [...result.layers].reverse(), [result.layers]);
  return (
    <section className="rank-heatmap-card">
      <header className="card-heading compact-heading">
        <div><span className="eyebrow">PINNED RANK MAP</span><h2>{tokenId == null ? "固定 token 全局排名" : <>rank of <strong>{tokenLabel(result.vocab[String(tokenId)] ?? `[${tokenId}]`, showWhitespace)}</strong></>}</h2></div>
        <Grid3X3 size={16} />
      </header>
      {!track ? (
        <div className="rank-empty"><Pin size={18} /><span>从 Top‑10 表或主网格中固定一个 token</span></div>
      ) : (
        <div className="rank-map-wrap">
          <div className="rank-map" style={{ gridTemplateColumns: `32px repeat(${result.tokens.length}, minmax(14px, 1fr))` }}>
            {reversedLayers.flatMap((layer) => {
              const rowIndex = layerIndex(result, layer);
              return [
                <span className={selectedLayer === layer ? "selected" : ""} key={`r-label-${layer}`}>{layer}</span>,
                ...track[rowIndex].map((rank, position) => (
                  <button
                    type="button"
                    key={`${layer}-${position}`}
                    className={selectedLayer === layer && selectedPosition === position ? "selected" : ""}
                    style={{ background: rankColor(rank, result.vocab_size) }}
                    aria-label={`Layer ${layer}, Position ${position}, rank ${rank}`}
                    title={`L${layer} · P${position} · rank #${rank}`}
                    onClick={() => onSelect(layer, position)}
                  />
                )),
              ];
            })}
          </div>
          <div className="rank-position-axis" style={{ gridTemplateColumns: `32px repeat(${result.tokens.length}, minmax(14px, 1fr))` }}>
            <span />{result.tokens.map((token) => <small key={token.index} className={selectedPosition === token.index ? "selected" : ""}>{token.index}</small>)}
          </div>
        </div>
      )}
    </section>
  );
}

function rankPath(values: number[], vocabSize: number, width = 360, height = 156) {
  const ceiling = Math.log(Math.max(2, vocabSize));
  return values.map((rank, index) => {
    const x = values.length <= 1 ? width / 2 : (index / (values.length - 1)) * width;
    const y = 8 + (Math.log(Math.max(1, rank)) / ceiling) * (height - 16);
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
}

function RankChart({
  title,
  axisLabel,
  series,
  vocabSize,
  selectedIndex,
}: {
  title: string;
  axisLabel: string;
  series: Array<{ id: number; label: string; values: number[]; color: string }>;
  vocabSize: number;
  selectedIndex: number;
}) {
  const width = 360;
  const height = 156;
  const length = series[0]?.values.length ?? 0;
  const markerX = length <= 1 ? width / 2 : (selectedIndex / (length - 1)) * width;
  return (
    <section className="rank-chart-card">
      <header className="card-heading compact-heading"><div><span className="eyebrow">RANK TRACE</span><h2>{title}</h2></div><ChartNoAxesCombined size={16} /></header>
      {series.length ? (
        <>
          <div className="rank-chart-body">
            <div className="rank-chart-y"><span>#1</span><span>1k</span><span>{compactRank(vocabSize)}</span></div>
            <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
              <line x1="0" x2={width} y1="8" y2="8" />
              <line x1="0" x2={width} y1={height / 2} y2={height / 2} />
              <line x1="0" x2={width} y1={height - 8} y2={height - 8} />
              <line className="selection-marker" x1={markerX} x2={markerX} y1="0" y2={height} />
              {series.map((item) => <path key={item.id} d={rankPath(item.values, vocabSize, width, height)} style={{ stroke: item.color }} />)}
              {series.map((item) => {
                const rank = item.values[selectedIndex] ?? vocabSize;
                const y = 8 + (Math.log(Math.max(1, rank)) / Math.log(Math.max(2, vocabSize))) * (height - 16);
                return <circle key={`dot-${item.id}`} cx={markerX} cy={y} r="3.5" style={{ fill: item.color }} />;
              })}
            </svg>
          </div>
          <div className="rank-chart-footer"><span>0</span><strong>{axisLabel}</strong><span>{Math.max(0, length - 1)}</span></div>
        </>
      ) : <div className="rank-empty"><Pin size={18} /><span>固定 token 后显示完整词表排名轨迹</span></div>}
    </section>
  );
}

export function OfficialExplorer({
  result,
  runState,
  selectedLayer,
  selectedPosition,
  pinned,
  activePinnedToken,
  showWhitespace,
  onSelect,
  onPin,
  onActivatePin,
  onClearPins,
  onShowWhitespaceChange,
}: ExplorerProps) {
  if (!result) {
    return (
      <main className="official-explorer empty-explorer">
        <div className={`explorer-placeholder ${runState === "running" ? "loading" : ""}`}>
          <ScanSearch size={28} />
          <strong>{runState === "running" ? "正在构建真实 J-lens slice" : "准备探索模型内部概念"}</strong>
          <span>{runState === "running" ? "Qwen 前向计算 → Jacobian 传输 → 完整词表排名" : "选择样本并开始真实探查，页面将生成官方式四联视图。"}</span>
          <div className="placeholder-grid" aria-hidden="true">{Array.from({ length: 96 }, (_, index) => <i key={index} />)}</div>
        </div>
      </main>
    );
  }

  const selectedLayerIndex = layerIndex(result, selectedLayer);
  const inputToken = result.tokens[selectedPosition];
  const tracked = [...pinned]
    .filter((tokenId) => result.rank_tracks[String(tokenId)])
    .slice(0, PALETTE.length)
    .map((tokenId, index) => ({
      id: tokenId,
      label: tokenLabel(result.vocab[String(tokenId)] ?? `[${tokenId}]`, showWhitespace),
      color: PALETTE[index],
    }));

  const byLayerRows = [...result.layers].reverse().map((layer) => ({
    key: `layer-${layer}`,
    label: String(layer),
    selected: layer === selectedLayer,
    candidates: result.cells[layerIndex(result, layer)][selectedPosition].candidates,
    onSelect: () => onSelect(layer, selectedPosition),
  }));
  const byPositionRows = result.tokens.map((token) => ({
    key: `position-${token.index}`,
    label: String(token.index),
    context: tokenLabel(token.text, showWhitespace),
    selected: token.index === selectedPosition,
    candidates: result.cells[selectedLayerIndex][token.index].candidates,
    onSelect: () => onSelect(selectedLayer, token.index),
  }));
  const layerSeries = tracked.map((item) => ({
    ...item,
    values: rankTrajectory(result, selectedPosition, item.id),
  }));
  const positionSeries = tracked.map((item) => ({
    ...item,
    values: result.rank_tracks[String(item.id)][selectedLayerIndex],
  }));
  const finalOutput = result.final_outputs[selectedPosition];

  return (
    <main className="official-explorer">
      <section className="selection-toolbar">
        <div className="selection-primary">
          <span>Position <strong>{selectedPosition}</strong></span>
          <code title={`token ${inputToken.id}`}>{tokenLabel(inputToken.text, showWhitespace)}</code>
          <i />
          <span>Layer <strong>{selectedLayer}</strong></span>
        </div>
        <label className="whitespace-toggle">
          <input type="checkbox" checked={showWhitespace} onChange={(event) => onShowWhitespaceChange(event.target.checked)} />
          whitespace
        </label>
        <span className="keyboard-hint"><Keyboard size={14} />hold ⇧ to scrub · ← → pos · ↑ ↓ layer</span>
      </section>

      <section className="slice-layout">
        <TokenMatrix
          result={result}
          selectedLayer={selectedLayer}
          selectedPosition={selectedPosition}
          pinned={pinned}
          showWhitespace={showWhitespace}
          onSelect={onSelect}
          onPin={onPin}
        />
        <div className="slice-tables">
          <SliceTable title={`By Layer · Pos ${selectedPosition}`} subtitle={tokenLabel(inputToken.text, showWhitespace)} rows={byLayerRows} pinned={pinned} showWhitespace={showWhitespace} onPin={onPin} />
          <SliceTable title={`By Position · Layer ${selectedLayer}`} subtitle={`${result.tokens.length} positions`} rows={byPositionRows} pinned={pinned} showWhitespace={showWhitespace} onPin={onPin} />
        </div>
      </section>

      <section className="prompt-readout">
        <div className="prompt-token-strip">
          {result.tokens.map((token) => (
            <button type="button" key={token.index} className={selectedPosition === token.index ? "selected" : ""} onClick={() => onSelect(selectedLayer, token.index)}>
              {tokenLabel(token.text, showWhitespace)}
            </button>
          ))}
        </div>
        <div className="final-output-readout">
          <span>最终层下一 token</span>
          <strong>{tokenLabel(finalOutput.top_token, showWhitespace)}</strong>
          <small>{(finalOutput.top_probability * 100).toFixed(2)}%</small>
        </div>
      </section>

      <PinnedTokens result={result} pinned={pinned} activePinnedToken={activePinnedToken} showWhitespace={showWhitespace} onPin={onPin} onActivatePin={onActivatePin} onClearPins={onClearPins} />

      <section className="rank-layout">
        <RankHeatmap result={result} tokenId={activePinnedToken} selectedLayer={selectedLayer} selectedPosition={selectedPosition} showWhitespace={showWhitespace} onSelect={onSelect} />
        <RankChart title={`By Layer · Pos ${selectedPosition}`} axisLabel="Layer →" series={layerSeries} vocabSize={result.vocab_size} selectedIndex={selectedLayerIndex} />
        <RankChart title={`By Position · Layer ${selectedLayer}`} axisLabel="Position →" series={positionSeries} vocabSize={result.vocab_size} selectedIndex={selectedPosition} />
      </section>

      <section className="official-footnote">
        <CircleHelp size={14} />
        <span>候选使用官方 <code>mask_display=True</code> 只过滤显示内容；所有红色上标、热力图和曲线均为未过滤完整词表排名。</span>
      </section>
    </main>
  );
}
