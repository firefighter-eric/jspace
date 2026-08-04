import { LoaderCircle, Play } from "lucide-react";
import { samples, type Sample } from "../data";
import type { RunState, RuntimeStatus } from "../types";

type Props = {
  activeSampleId: string | null;
  prompt: string;
  runState: RunState;
  runtimeStatus: RuntimeStatus | null;
  tokenCount: number | null;
  truncated: boolean;
  elapsedMs: number | null;
  error: string | null;
  onPromptChange: (value: string) => void;
  onSelectSample: (sample: Sample) => void;
  onRun: () => void;
};

export function PromptComposer({
  activeSampleId,
  prompt,
  runState,
  runtimeStatus,
  tokenCount,
  truncated,
  elapsedMs,
  error,
  onPromptChange,
  onSelectSample,
  onRun,
}: Props) {
  const running = runState === "running";
  return (
    <section className="prompt-composer" aria-label="整句提示词输入">
      <div className="composer-label">
        <strong>Prompt</strong>
        <span>Enter a full sentence.<br />Analyze each token × layer.</span>
      </div>
      <textarea
        aria-label="完整提示词"
        value={prompt}
        rows={2}
        disabled={running}
        onChange={(event) => onPromptChange(event.target.value)}
        onKeyDown={(event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "Enter") onRun();
        }}
      />
      <div className="composer-actions">
        <button className="run-action" type="button" disabled={running || !prompt.trim()} onClick={onRun}>
          {running ? <LoaderCircle className="spin" size={14} /> : <Play size={13} fill="currentColor" />}
          {running ? "Analyzing…" : "Run J-lens"}
        </button>
      </div>
      <div className="composer-examples" aria-label="Anthropic evaluation examples">
        <span>Official examples</span>
        <div>
          {samples.map((sample) => (
            <button
              type="button"
              key={sample.id}
              aria-pressed={activeSampleId === sample.id}
              disabled={running}
              title={sample.prompt}
              onClick={() => onSelectSample(sample)}
            >
              {sample.name}
              <small>{sample.target}</small>
            </button>
          ))}
        </div>
      </div>
      <div className="composer-meta">
        <div className="runtime-inline">
          <span className={`runtime-light ${runtimeStatus?.state ?? "unknown"}`} />
          {runtimeStatus?.state === "ready" ? "model ready" : runtimeStatus?.state === "loading" ? "loading model" : "local runtime"}
          {tokenCount != null ? ` · ${tokenCount} positions × 32 layers` : ""}
          {elapsedMs != null ? ` · ${(elapsedMs / 1000).toFixed(2)}s` : ""}
          {truncated && tokenCount != null ? (
            <strong className="truncation-warning"> · input truncated after {tokenCount} positions</strong>
          ) : null}
        </div>
      </div>
      {error ? <p className="composer-error" role="alert">{error}</p> : null}
    </section>
  );
}
