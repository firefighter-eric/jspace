"""J-Space API semantics against the official slice conventions."""

from __future__ import annotations

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
