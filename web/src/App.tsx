import { useCallback, useEffect, useState } from "react";
import { analyzePrompt, fetchRuntimeStatus } from "./api";
import { layerIndex, normalizedToken, PALETTE } from "./analysis";
import { OfficialExplorer } from "./components/OfficialExplorer";
import { PromptComposer } from "./components/PromptComposer";
import { samples } from "./data";
import type { AnalysisResult, RunState, RuntimeStatus } from "./types";

function App() {
  const [activeSampleId, setActiveSampleId] = useState<string | null>(samples[0].id);
  const [prompt, setPrompt] = useState(samples[0].prompt);
  const [runState, setRunState] = useState<RunState>("idle");
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [selectedLayer, setSelectedLayer] = useState(16);
  const [selectedPosition, setSelectedPosition] = useState(0);
  const [pinned, setPinned] = useState<Set<number>>(() => new Set());
  const [activePinnedToken, setActivePinnedToken] = useState<number | null>(null);
  const [showWhitespace, setShowWhitespace] = useState(false);
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
    setActivePinnedToken(null);
    setSelectedLayer(16);
    setSelectedPosition(0);
  }, []);

  function handlePromptChange(nextPrompt: string) {
    setPrompt(nextPrompt);
    const matchingSample = samples.find((sample) => sample.prompt === nextPrompt);
    setActiveSampleId(matchingSample?.id ?? null);
    clearResult();
  }

  async function handleRun() {
    if (!prompt.trim() || runState === "running") return;
    setAnalysis(null);
    setPinned(new Set());
    setActivePinnedToken(null);
    setError(null);
    setRunState("running");
    void refreshStatus();
    try {
      const result = await analyzePrompt(prompt);
      const initialLayer = result.layers.includes(16) ? 16 : result.default_selection.layer;
      setAnalysis(result);
      setSelectedLayer(initialLayer);
      setSelectedPosition(result.default_selection.position);
      const activeSample = samples.find((sample) => sample.id === activeSampleId);
      const targetIds = activeSample
        ? result.tracked_token_ids
            .filter((tokenId) => normalizedToken(result.vocab[String(tokenId)] ?? "") === normalizedToken(activeSample.target))
            .sort((left, right) => Number(!(result.vocab[String(left)] ?? "").startsWith(" ")) - Number(!(result.vocab[String(right)] ?? "").startsWith(" ")))
        : [];
      const defaultCell = result.cells[layerIndex(result, initialLayer)][result.default_selection.position];
      const defaultPin = targetIds[0] ?? (result.tracked_token_ids.includes(defaultCell.top_id) ? defaultCell.top_id : null);
      setPinned(defaultPin == null ? new Set() : new Set([defaultPin]));
      setActivePinnedToken(defaultPin);
      setRunState("success");
    } catch (analysisError) {
      setError(analysisError instanceof Error ? analysisError.message : String(analysisError));
      setRunState("error");
    } finally {
      void refreshStatus();
    }
  }

  function handleSelect(layer: number, position: number) {
    setSelectedLayer(layer);
    setSelectedPosition(position);
  }

  function togglePin(tokenId: number) {
    if (!analysis?.tracked_token_ids.includes(tokenId)) return;
    setPinned((current) => {
      const next = new Set(current);
      if (next.has(tokenId)) {
        next.delete(tokenId);
        setActivePinnedToken((active) => active === tokenId ? (next.values().next().value ?? null) : active);
      } else if (next.size < PALETTE.length) {
        next.add(tokenId);
        setActivePinnedToken(tokenId);
      }
      return next;
    });
  }

  function activatePin(tokenId: number) {
    if (!analysis?.tracked_token_ids.includes(tokenId)) return;
    setPinned((current) => {
      if (current.has(tokenId)) return current;
      const next = new Set(current);
      if (next.size < PALETTE.length) next.add(tokenId);
      return next;
    });
    setActivePinnedToken(tokenId);
  }

  function clearPins() {
    setPinned(new Set());
    setActivePinnedToken(null);
  }

  useEffect(() => {
    if (!analysis) return;
    const result = analysis;
    function handleKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target?.matches("textarea, input, select, [contenteditable='true']")) return;
      const currentLayerIndex = layerIndex(result, selectedLayer);
      if (event.key === "ArrowLeft") setSelectedPosition((position) => Math.max(0, position - 1));
      else if (event.key === "ArrowRight") setSelectedPosition((position) => Math.min(result.tokens.length - 1, position + 1));
      else if (event.key === "ArrowUp") setSelectedLayer(result.layers[Math.min(result.layers.length - 1, currentLayerIndex + 1)]);
      else if (event.key === "ArrowDown") setSelectedLayer(result.layers[Math.max(0, currentLayerIndex - 1)]);
      else return;
      event.preventDefault();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [analysis, selectedLayer]);

  return (
    <div className="anth-app">
      <header className="anth-header">
        <div>
          <h1>Anthropic Jacobian Lens <span>— Official-style Interactive UI</span></h1>
          <p>Qwen3.5-4B · n1000 · official mask_display=True · full-sentence input</p>
        </div>
        <a href="https://github.com/anthropics/jacobian-lens" target="_blank" rel="noreferrer" aria-label="Anthropic 官方仓库">?</a>
      </header>

      <PromptComposer
          prompt={prompt}
          runState={runState}
          runtimeStatus={runtimeStatus}
          tokenCount={analysis?.tokens.length ?? null}
          elapsedMs={analysis?.elapsed_ms ?? null}
          error={error}
          onPromptChange={handlePromptChange}
          onRun={() => void handleRun()}
        />
      <div className="official-shell">
        <OfficialExplorer
          result={analysis}
          runState={runState}
          selectedLayer={selectedLayer}
          selectedPosition={selectedPosition}
          pinned={pinned}
          activePinnedToken={activePinnedToken}
          showWhitespace={showWhitespace}
          onSelect={handleSelect}
          onPin={togglePin}
          onActivatePin={activatePin}
          onClearPins={clearPins}
          onShowWhitespaceChange={setShowWhitespace}
        />
      </div>
      <footer className="anth-footer">
        Anthropic Jacobian Lens — local interactive slice · {analysis ? `${analysis.tokens.length} positions × ${analysis.layers.length} layers · top-10` : "waiting for analysis"}
      </footer>
    </div>
  );
}

export default App;
