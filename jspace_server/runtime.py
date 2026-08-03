"""Lazy local Qwen + Jacobian-lens inference runtime.

The browser must never invent semantic candidates.  This module is the single
source of truth for tokenizer positions, per-cell candidates, probabilities,
and provenance shown by the Observatory UI.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import transformers

import jlens
from jlens.hooks import ActivationRecorder
from jlens.vis import _meaningful_token_mask, _ranks_of

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = Path("/Users/eric/models/Qwen/Qwen3.5-4B")
DEFAULT_LENS_PATH = (
    PROJECT_ROOT / "artifacts/lenses/Qwen3.5-4B_jacobian_lens_n1000.pt"
)


def _device_name() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class AnalysisRuntime:
    """Load the local model once and serialize true J-lens readouts."""

    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        lens_path: Path = DEFAULT_LENS_PATH,
    ) -> None:
        self.model_path = model_path
        self.lens_path = lens_path
        self.device = _device_name()
        self._state = "not_loaded"
        self._error: str | None = None
        self._loaded_at: float | None = None
        self._load_lock = threading.Lock()
        self._analysis_lock = threading.Lock()
        self._model: jlens.HFLensModel | None = None
        self._lens: jlens.JacobianLens | None = None

    def status(self) -> dict[str, Any]:
        return {
            "state": self._state,
            "error": self._error,
            "device": self.device,
            "model_path": str(self.model_path),
            "model_present": (self.model_path / "config.json").is_file(),
            "lens_path": str(self.lens_path),
            "lens_present": self.lens_path.is_file(),
            "loaded_at": self._loaded_at,
        }

    def ensure_loaded(self) -> None:
        if self._state == "ready":
            return
        with self._load_lock:
            if self._state == "ready":
                return
            if not (self.model_path / "config.json").is_file():
                raise FileNotFoundError(f"model not found: {self.model_path}")
            if not self.lens_path.is_file():
                raise FileNotFoundError(f"lens not found: {self.lens_path}")

            self._state = "loading"
            self._error = None
            try:
                dtype = torch.bfloat16
                hf_model = transformers.AutoModelForCausalLM.from_pretrained(
                    str(self.model_path),
                    dtype=dtype,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                )
                hf_model.to(self.device)
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    str(self.model_path), local_files_only=True
                )
                model = jlens.from_hf(hf_model, tokenizer)
                lens = jlens.JacobianLens.load(str(self.lens_path))
                if lens.d_model != model.d_model:
                    raise ValueError(
                        f"lens d_model={lens.d_model} does not match "
                        f"model d_model={model.d_model}"
                    )

                # Keep the fitted transforms on the compute device.  Leaving
                # them on CPU would transfer ~800 MB for every analysis.
                lens.jacobians = {
                    layer: jacobian.to(self.device)
                    for layer, jacobian in lens.jacobians.items()
                }
                self._model = model
                self._lens = lens
                self._loaded_at = time.time()
                self._state = "ready"
                if self.device == "mps":
                    torch.mps.empty_cache()
            except Exception as exc:
                self._state = "error"
                self._error = f"{type(exc).__name__}: {exc}"
                raise

    def analyze(
        self,
        prompt: str,
        *,
        top_k: int = 8,
        max_tokens: int = 64,
    ) -> dict[str, Any]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        if len(prompt) > 4000:
            raise ValueError("prompt is too long (maximum 4000 characters)")
        if not 2 <= top_k <= 12:
            raise ValueError("top_k must be between 2 and 12")
        if not 4 <= max_tokens <= 128:
            raise ValueError("max_tokens must be between 4 and 128")

        self.ensure_loaded()
        assert self._model is not None
        assert self._lens is not None

        with self._analysis_lock:
            started = time.perf_counter()
            result = self._analyze_loaded(prompt, top_k=top_k, max_tokens=max_tokens)
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            return result

    @torch.inference_mode()
    def _analyze_loaded(
        self, prompt: str, *, top_k: int, max_tokens: int
    ) -> dict[str, Any]:
        assert self._model is not None
        assert self._lens is not None
        model = self._model
        lens = self._lens
        tokenizer = model.tokenizer

        final_layer = model.n_layers - 1
        layers = sorted(set(lens.source_layers) | {final_layer})
        input_ids = model.encode(prompt, max_length=max_tokens)
        token_ids = input_ids[0].tolist()
        token_text = [
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            for token_id in token_ids
        ]

        activations: dict[int, torch.Tensor] = {}
        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)
            for layer in layers:
                activations[layer] = recorder.activations[layer].detach()

        def lens_logits(layer: int) -> torch.Tensor:
            residual = activations[layer][0].float()
            if layer in lens.jacobians:
                residual = lens.transport(residual, layer)
            return model.unembed(residual).float().detach()

        # Match Anthropic's Qwen walkthrough: choose display candidates with
        # ``mask_display=True`` semantics, but retain their true full-vocab
        # ranks and probabilities.  The unfiltered final-layer distribution is
        # returned separately so the UI never confuses a readable lens token
        # with the model's literal next-token prediction.
        rows: list[list[dict[str, Any]]] = []
        score_by_token: dict[int, float] = {}
        vocab_ids = set(token_ids)
        vocab_size = 0
        display_mask: torch.Tensor | None = None
        final_outputs: list[dict[str, Any]] = []
        previous_top_ids: list[int] | None = None
        for layer in layers:
            logits = lens_logits(layer)
            vocab_size = int(logits.shape[-1])
            if display_mask is None:
                display_mask = _meaningful_token_mask(
                    tokenizer, vocab_size, logits.device
                )

            top_values, top_ids = logits.masked_fill(
                ~display_mask, float("-inf")
            ).topk(top_k, dim=-1)
            top_ranks = _ranks_of(logits, top_ids) + 1
            log_partition = torch.logsumexp(logits, dim=-1, keepdim=True)
            top_probabilities = torch.exp(top_values - log_partition)
            logit_gaps = top_values[:, 0] - top_values[:, 1]

            ids_cpu = top_ids.cpu().tolist()
            values_cpu = top_values.cpu().tolist()
            ranks_cpu = top_ranks.cpu().tolist()
            probabilities_cpu = top_probabilities.cpu().tolist()
            gaps_cpu = logit_gaps.cpu().tolist()
            row: list[dict[str, Any]] = []
            current_top_ids: list[int] = []
            for position, candidate_ids in enumerate(ids_cpu):
                candidates = []
                for rank, token_id in enumerate(candidate_ids):
                    full_rank = ranks_cpu[position][rank]
                    score_by_token[token_id] = score_by_token.get(token_id, 0.0) + (
                        1.0 / full_rank
                    )
                    vocab_ids.add(token_id)
                    candidates.append(
                        {
                            "id": token_id,
                            "token": tokenizer.decode(
                                [token_id], clean_up_tokenization_spaces=False
                            ),
                            "rank": full_rank,
                            "probability": probabilities_cpu[position][rank],
                            "logit": values_cpu[position][rank],
                        }
                    )
                top_id = candidate_ids[0]
                current_top_ids.append(top_id)
                row.append(
                    {
                        "top_id": top_id,
                        "top_token": candidates[0]["token"],
                        "top_rank": candidates[0]["rank"],
                        "top_probability": candidates[0]["probability"],
                        "logit_gap": gaps_cpu[position],
                        "changed": (
                            previous_top_ids is not None
                            and previous_top_ids[position] != top_id
                        ),
                        "candidates": candidates,
                    }
                )
            rows.append(row)
            previous_top_ids = current_top_ids

            if layer == final_layer:
                raw_values, raw_ids = logits.topk(top_k, dim=-1)
                raw_probabilities = torch.exp(raw_values - log_partition)
                raw_ids_cpu = raw_ids.cpu().tolist()
                raw_values_cpu = raw_values.cpu().tolist()
                raw_probabilities_cpu = raw_probabilities.cpu().tolist()
                for position, candidate_ids in enumerate(raw_ids_cpu):
                    candidates = []
                    for raw_rank, token_id in enumerate(candidate_ids, start=1):
                        vocab_ids.add(token_id)
                        candidates.append(
                            {
                                "id": token_id,
                                "token": tokenizer.decode(
                                    [token_id], clean_up_tokenization_spaces=False
                                ),
                                "rank": raw_rank,
                                "probability": raw_probabilities_cpu[position][
                                    raw_rank - 1
                                ],
                                "logit": raw_values_cpu[position][raw_rank - 1],
                            }
                        )
                    final_outputs.append(
                        {
                            "top_id": candidate_ids[0],
                            "top_token": candidates[0]["token"],
                            "top_probability": candidates[0]["probability"],
                            "candidates": candidates,
                        }
                    )
                del raw_values, raw_ids, raw_probabilities

            del logits, top_values, top_ids, top_ranks, top_probabilities

        # Anthropic's default slice path tracks every token that appears in any
        # readable Top-K cell and loads its full-vocab rank tensor when pinned.
        by_score = sorted(score_by_token, key=score_by_token.__getitem__, reverse=True)
        tracked_token_ids = sorted(by_score)
        rank_tracks: dict[str, list[list[int]]] = {
            str(token_id): [] for token_id in tracked_token_ids
        }
        if tracked_token_ids:
            tracked_tensor = torch.tensor(
                tracked_token_ids, dtype=torch.long, device=model.input_device
            )
            for layer in layers:
                logits = lens_logits(layer)
                rank_matrix = (_ranks_of(logits, tracked_tensor) + 1).cpu().tolist()
                for tracked_index, token_id in enumerate(tracked_token_ids):
                    rank_tracks[str(token_id)].append(
                        [row[tracked_index] for row in rank_matrix]
                    )
                del logits, rank_matrix

        vocab_ids.update(tracked_token_ids)
        vocab = {
            str(token_id): tokenizer.decode(
                [token_id], clean_up_tokenization_spaces=False
            )
            for token_id in vocab_ids
        }

        activations.clear()
        if self.device == "mps":
            torch.mps.synchronize()

        default_layer = min(20, layers[-1])
        return {
            "provenance": "real-jacobian-lens",
            "prompt": prompt,
            "model": "Qwen/Qwen3.5-4B",
            "model_path": str(self.model_path),
            "lens": self.lens_path.name,
            "lens_prompts": lens.n_prompts,
            "device": self.device,
            "display_mode": "wordlike-full-vocab-rank",
            "vocab_size": vocab_size,
            "layers": layers,
            "tokens": [
                {"index": index, "id": token_id, "text": text}
                for index, (token_id, text) in enumerate(
                    zip(token_ids, token_text, strict=True)
                )
            ],
            "cells": rows,
            "final_outputs": final_outputs,
            "tracked_token_ids": tracked_token_ids,
            "rank_tracks": rank_tracks,
            "vocab": vocab,
            "default_selection": {
                "layer": default_layer,
                "position": max(0, len(token_ids) - 1),
            },
            "truncated": len(token_ids) >= max_tokens,
        }
