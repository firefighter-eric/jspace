import { useCallback, useEffect, useState } from "react";
import { analyzePrompt, fetchRuntimeStatus } from "./api";
import { PALETTE, selectedCell } from "./analysis";
import { HeatmapPanel } from "./components/HeatmapPanel";
import { InspectorPanel } from "./components/InspectorPanel";
import { PromptPanel } from "./components/PromptPanel";
import { TracePanels } from "./components/TracePanels";
import { samples, type Sample } from "./data";
import type { AnalysisResult, Metric, RunState, RuntimeStatus } from "./types";

function App() {
  const [activeSampleId, setActiveSampleId] = useState<string | null>(samples[0].id);
  const [prompt, setPrompt] = useState(samples[0].prompt);
  const [metric, setMetric] = useState<Metric>("rank");
  const [runState, setRunState] = useState<RunState>("idle");
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [selectedLayer, setSelectedLayer] = useState(20);
  const [selectedPosition, setSelectedPosition] = useState(0);
  const [pinned, setPinned] = useState<Set<number>>(() => new Set());
  const [error, setError] = useState<string | null>(null);

  const refreshStatus = useCallback(async (signal?: AbortSignal) => {
    try {
      setRuntimeStatus(await fetchRuntimeStatus(signal));
    } catch (statusError) {
      if (!(statusError instanceof DOMException && statusError.name === "AbortError")) {
        setRuntimeStatus(null);
      }
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refreshStatus(controller.signal);
    return () => controller.abort();
  }, [refreshStatus]);

  const clearResult = useCallback(() => {
    setAnalysis(null);
    setRunState("idle");
    setError(null);
    setPinned(new Set());
    setSelectedLayer(20);
    setSelectedPosition(0);
  }, []);

  function handlePromptChange(nextPrompt: string) {
    setPrompt(nextPrompt);
    const matchingSample = samples.find((sample) => sample.prompt === nextPrompt);
    setActiveSampleId(matchingSample?.id ?? null);
    clearResult();
  }

  function handleSelectSample(sample: Sample) {
    setActiveSampleId(sample.id);
    setPrompt(sample.prompt);
    clearResult();
  }

  async function handleRun() {
    if (!prompt.trim() || runState === "running") return;
    setAnalysis(null);
    setPinned(new Set());
    setError(null);
    setRunState("running");
    void refreshStatus();
    try {
      const result = await analyzePrompt(prompt);
      setAnalysis(result);
      setSelectedLayer(result.default_selection.layer);
      setSelectedPosition(result.default_selection.position);
      setPinned(new Set());
      setRunState("success");
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : String(analysisError));
      setRunState("error");
    } finally {
      void refreshStatus();
    }
  }

  function handleSelect(layer: number, position: number, tokenId?: number) {
    setSelectedLayer(layer);
    setSelectedPosition(position);
    if (tokenId !== undefined) togglePin(tokenId);
  }

  function togglePin(tokenId: number) {
    if (!analysis?.tracked_token_ids.includes(tokenId)) return;
    setPinned((current) => {
      const next = new Set(current);
      if (next.has(tokenId)) next.delete(tokenId);
      else if (next.size < PALETTE.length) next.add(tokenId);
      return next;
    });
  }

  function exportCell() {
    if (!analysis) return;
    const cell = selectedCell(analysis, selectedLayer, selectedPosition);
    const payload = {
      provenance: analysis.provenance,
      model: analysis.model,
      lens: analysis.lens,
      prompt: analysis.prompt,
      layer: selectedLayer,
      position: selectedPosition,
      input_token: analysis.tokens[selectedPosition],
      candidates: cell.candidates,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `jspace-real-L${selectedLayer}-P${selectedPosition}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }

  const headerStatus = runState === "running"
    ? "Qwen + J-lens 分析中"
    : analysis
      ? `真实读数 · ${analysis.tokens.length} tokens`
      : runtimeStatus?.state === "ready"
        ? "本地模型已加载"
        : "本地模型与 Lens 已就绪";

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">J</span>
          <div><strong>J-Space Observatory</strong><span className="brand-subtitle">Jacobian Lens Explorer</span></div>
        </div>
        <div className="topbar-status"><span className={`status-dot ${runState === "running" ? "loading" : analysis ? "ready" : "not_loaded"}`} />{headerStatus}</div>
        <div className="topbar-controls">
          <div className="select-control static"><span>模型</span><span>Qwen3.5-4B</span></div>
          <div className="select-control static"><span>透镜</span><span>J-lens · n1000</span></div>
        </div>
      </header>

      <div className="workspace">
        <PromptPanel
          activeSampleId={activeSampleId}
          prompt={prompt}
          runState={runState}
          runtimeStatus={runtimeStatus}
          tokenCount={analysis?.tokens.length ?? null}
          elapsedMs={analysis?.elapsed_ms ?? null}
          error={error}
          onPromptChange={handlePromptChange}
          onResetPrompt={() => handleSelectSample(samples[0])}
          onSelectSample={handleSelectSample}
          onRun={() => void handleRun()}
        />
        <main className="analysis">
          <HeatmapPanel
            result={analysis}
            metric={metric}
            selectedLayer={selectedLayer}
            selectedPosition={selectedPosition}
            pinned={pinned}
            runState={runState}
            onMetricChange={setMetric}
            onSelect={handleSelect}
          />
          <TracePanels result={analysis} selectedLayer={selectedLayer} selectedPosition={selectedPosition} pinned={pinned} />
        </main>
        <InspectorPanel
          result={analysis}
          metric={metric}
          selectedLayer={selectedLayer}
          selectedPosition={selectedPosition}
          pinned={pinned}
          onTogglePin={togglePin}
          onExport={exportCell}
        />
      </div>

      <footer className="statusbar">
        <span><span className={`status-dot ${analysis ? "ready" : "not_loaded"}`} />{analysis ? "real analysis ready" : "waiting for analysis"}</span>
        <span>model · /Users/eric/models/Qwen/Qwen3.5-4B</span>
        <span>lens · Qwen3.5-4B_jacobian_lens_n1000.pt</span>
        <span>mode · {analysis ? "real-jacobian-lens" : "no result"}</span>
      </footer>
    </div>
  );
}

export default App;
