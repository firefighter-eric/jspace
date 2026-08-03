import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  Check,
  Info,
  LoaderCircle,
  Play,
  RotateCcw,
} from "lucide-react";
import { samples, type Sample } from "../data";
import type { RunState, RuntimeStatus } from "../types";

type Props = {
  activeSampleId: string | null;
  prompt: string;
  runState: RunState;
  runtimeStatus: RuntimeStatus | null;
  tokenCount: number | null;
  elapsedMs: number | null;
  error: string | null;
  onPromptChange: (prompt: string) => void;
  onResetPrompt: () => void;
  onSelectSample: (sample: Sample) => void;
  onRun: () => void;
};

function runtimeLabel(status: RuntimeStatus | null) {
  if (!status) return "检查后端…";
  if (status.state === "ready") return `模型就绪 · ${status.device.toUpperCase()}`;
  if (status.state === "loading") return "正在加载模型";
  if (status.state === "error") return "模型加载失败";
  if (!status.model_present || !status.lens_present) return "模型文件缺失";
  return "本地文件就绪";
}

export function PromptPanel({
  activeSampleId,
  prompt,
  runState,
  runtimeStatus,
  tokenCount,
  elapsedMs,
  error,
  onPromptChange,
  onResetPrompt,
  onSelectSample,
  onRun,
}: Props) {
  const running = runState === "running";
  return (
    <aside className="prompt-panel">
      <section>
        <div className="section-heading">
          <div><span className="eyebrow">INPUT</span><h2>提示词</h2></div>
          <button className="icon-button small" type="button" aria-label="重置提示词" onClick={onResetPrompt}><RotateCcw size={15} /></button>
        </div>
        <textarea aria-label="提示词输入" value={prompt} onChange={(event) => onPromptChange(event.target.value)} />
        <div className="editor-meta">
          <span>{prompt.length} chars</span>
          <span>{activeSampleId ? "官方评测输入" : "自定义输入"}</span>
        </div>
      </section>

      <section className="samples-section">
        <div className="section-heading compact"><div><span className="eyebrow">DATASET</span><h2>样本</h2></div><span className="count">{samples.length}</span></div>
        <div className="sample-list">
          {samples.map((item, index) => (
            <button
              key={item.id}
              className={`sample-row ${item.id === activeSampleId ? "selected" : ""}`}
              onClick={() => onSelectSample(item)}
              type="button"
            >
              <span className="sample-index">0{index + 1}</span>
              <span><strong>{item.name}</strong><small>{item.category}</small></span>
              {item.id === activeSampleId ? <Check size={16} /> : <ArrowUpRight size={14} />}
            </button>
          ))}
        </div>
      </section>

      <button className="run-button" type="button" onClick={onRun} disabled={running || !prompt.trim()}>
        {running ? <LoaderCircle className="spin" size={18} /> : <Play size={17} fill="currentColor" />}
        {running ? "Qwen + J-lens 分析中…" : "开始真实探查"}
      </button>

      {error ? (
        <div className="notice error" role="alert"><AlertTriangle size={16} /><p><strong>分析失败</strong>{error}</p></div>
      ) : (
        <div className={`notice ${runState === "success" ? "success" : ""}`} role="note">
          <Info size={16} />
          {runState === "success" ? (
            <p><strong>真实 J-lens 结果</strong>所有位置、候选词和概率均来自本地 Qwen3.5-4B 的同一次前向计算。</p>
          ) : running ? (
            <p><strong>正在运行真实模型</strong>首次请求包含模型加载；请保持页面打开，结果返回前不会展示旧候选。</p>
          ) : (
            <p><strong>等待真实分析</strong>输入变化后旧结果会立即清空；点击按钮后才显示模型候选。</p>
          )}
        </div>
      )}

      <div className="runtime">
        <div><span className={`status-dot ${runtimeStatus?.state ?? "unknown"}`} />{runtimeLabel(runtimeStatus)}</div>
        <div><Activity size={14} />{tokenCount == null ? "等待 tokenizer" : `32 层 · ${tokenCount} tokens`}</div>
      </div>
      {elapsedMs != null ? <div className="elapsed">本次分析 {(elapsedMs / 1000).toFixed(2)} s</div> : null}
    </aside>
  );
}
