export type Metric = "rank" | "probability";
export type RunState = "idle" | "running" | "success" | "error";
export type VocabularyMode = "readable" | "raw";

export type RuntimeStatus = {
  state: "not_loaded" | "loading" | "ready" | "error";
  error: string | null;
  device: string;
  model_path: string;
  model_present: boolean;
  lens_path: string;
  lens_present: boolean;
  loaded_at: number | null;
};

export type TokenInfo = {
  index: number;
  id: number;
  text: string;
};

export type Candidate = {
  id: number;
  token: string;
  rank: number;
  probability: number;
  logit: number;
};

export type AnalysisCell = {
  top_id: number;
  top_token: string;
  top_rank: number;
  top_probability: number;
  logit_gap: number;
  changed: boolean;
  candidates: Candidate[];
  raw_candidates?: Candidate[];
};

export type FinalOutput = {
  top_id: number;
  top_token: string;
  top_probability: number;
  candidates: Candidate[];
};

export type AnalysisResult = {
  provenance: "real-jacobian-lens";
  prompt: string;
  model: string;
  model_path: string;
  lens: string;
  lens_prompts: number;
  device: string;
  elapsed_ms: number;
  display_mode: "wordlike-full-vocab-rank";
  vocab_size: number;
  layers: number[];
  tokens: TokenInfo[];
  cells: AnalysisCell[][];
  final_outputs: FinalOutput[];
  tracked_token_ids: number[];
  rank_tracks: Record<string, number[][]>;
  rank_tracks_truncated: boolean;
  max_tracked_tokens: number;
  vocab: Record<string, string>;
  default_selection: {
    layer: number;
    position: number;
  };
  truncated: boolean;
};
