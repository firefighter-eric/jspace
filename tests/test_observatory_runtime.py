"""J-Space API semantics against the official slice conventions."""

from __future__ import annotations

import pytest
import torch

from jlens.lens import JacobianLens
from jlens.vis import _ranks_of
from jspace_server.runtime import AnalysisRuntime

from .tiny import TinyDecoder


def _runtime_with_tiny_model() -> tuple[AnalysisRuntime, TinyDecoder, JacobianLens]:
    model = TinyDecoder(n_layers=4, d_model=8, vocab_size=32)
    lens = JacobianLens(
        {layer: torch.eye(model.d_model) for layer in range(model.n_layers - 1)},
        n_prompts=7,
        d_model=model.d_model,
    )
    runtime = AnalysisRuntime()
    runtime.device = "cpu"
    runtime._model = model
    runtime._lens = lens
    runtime._state = "ready"
    return runtime, model, lens


def test_observatory_uses_wordlike_display_tokens_and_full_vocab_ranks():
    runtime, _, _ = _runtime_with_tiny_model()
    result = runtime._analyze_loaded("the quick brown fox", top_k=4, max_tokens=32)

    assert result["display_mode"] == "wordlike-full-vocab-rank"
    assert result["vocab_size"] == 32
    for row in result["cells"]:
        for cell in row:
            for candidate in cell["candidates"]:
                token = candidate["token"].strip()
                assert token
                assert all(
                    char.isalnum()
                    or (0 < index < len(token) - 1 and char in "'-’")
                    for index, char in enumerate(token)
                )
                assert candidate["rank"] >= 1


def test_observatory_final_output_and_rank_tracks_are_unfiltered_and_exact():
    runtime, model, lens = _runtime_with_tiny_model()
    prompt = "the quick brown fox"
    result = runtime._analyze_loaded(prompt, top_k=4, max_tokens=32)

    _, model_logits, _ = lens.apply(model, prompt, layers=[0])
    expected_top_ids = model_logits.argmax(dim=-1).tolist()
    assert [cell["top_id"] for cell in result["final_outputs"]] == expected_top_ids

    token_id = result["tracked_token_ids"][0]
    expected_ranks = (
        _ranks_of(model_logits, torch.tensor([token_id]))[:, 0] + 1
    ).tolist()
    final_layer_index = result["layers"].index(model.n_layers - 1)
    assert result["rank_tracks"][str(token_id)][final_layer_index] == expected_ranks


def test_observatory_exposes_literal_raw_top_k_for_every_cell():
    runtime, model, lens = _runtime_with_tiny_model()
    prompt = "the quick brown fox"
    top_k = 4
    result = runtime._analyze_loaded(prompt, top_k=top_k, max_tokens=32)

    lens_logits, model_logits, _ = lens.apply(
        model, prompt, layers=lens.source_layers
    )
    for layer_index, layer in enumerate(result["layers"]):
        expected_logits = (
            model_logits if layer == model.n_layers - 1 else lens_logits[layer]
        )
        expected_ids = expected_logits.topk(top_k, dim=-1).indices.tolist()
        for position, cell in enumerate(result["cells"][layer_index]):
            raw_candidates = cell["raw_candidates"]
            assert [candidate["id"] for candidate in raw_candidates] == expected_ids[
                position
            ]
            assert [candidate["rank"] for candidate in raw_candidates] == list(
                range(1, top_k + 1)
            )
            assert all(
                candidate["id"] in result["tracked_token_ids"]
                for candidate in raw_candidates
            )

    assert [
        candidate["id"] for candidate in result["final_outputs"][0]["candidates"]
    ] == [
        candidate["id"]
        for candidate in result["cells"][-1][0]["raw_candidates"]
    ]


def test_observatory_preserves_prompt_verbatim_and_reports_exact_truncation():
    runtime, model, _ = _runtime_with_tiny_model()

    exact = runtime.analyze("abc", top_k=4, max_tokens=5)
    assert exact["prompt"] == "abc"
    assert exact["truncated"] is False
    assert [token["id"] for token in exact["tokens"]] == model.encode(
        "abc", max_length=6
    )[0].tolist()

    trailing = runtime.analyze("abc ", top_k=4, max_tokens=5)
    assert trailing["prompt"] == "abc "
    assert trailing["truncated"] is False
    assert [token["id"] for token in trailing["tokens"]] == model.encode(
        "abc ", max_length=6
    )[0].tolist()
    assert len(trailing["tokens"]) == len(exact["tokens"]) + 1

    truncated = runtime.analyze("abcd", top_k=4, max_tokens=4)
    assert truncated["truncated"] is True

    with pytest.raises(ValueError, match="must not be empty"):
        runtime.analyze("   ", top_k=4, max_tokens=4)


def test_observatory_caps_eager_rank_tracks(monkeypatch):
    runtime, _, _ = _runtime_with_tiny_model()
    monkeypatch.setattr("jspace_server.runtime.MAX_TRACKED_TOKENS", 2)

    result = runtime._analyze_loaded("the quick brown fox", top_k=4, max_tokens=32)

    assert len(result["tracked_token_ids"]) == 2
    assert len(result["rank_tracks"]) == 2
    assert result["rank_tracks_truncated"] is True
    assert result["max_tracked_tokens"] == 2
