import pytest

from vllm_ascend_layered_prefill import (
    LAYERED_PREFILL_MODEL_ADAPTERS,
    LayeredPrefillMetadata,
    LayeredPrefillRequestData,
    get_layer_stage_range,
    get_moe_layer_cursors,
)


def test_stage_ranges_cover_model_once() -> None:
    ranges = [get_layer_stage_range(32, 3, stage) for stage in range(3)]
    assert ranges[0][0] == 0 and ranges[-1][1] == 32
    assert all(
        left[1] == right[0] for left, right in zip(ranges, ranges[1:], strict=False)
    )


def test_moe_cursors_cover_dense_prefix_and_multiple_experts() -> None:
    names = (
        "model.layers.2.mlp.experts",
        "model.layers.4.mlp.experts",
        "model.layers.4.shared_expert.experts",
    )
    assert get_moe_layer_cursors(names, 0, 6) == (0, 0, 0, 1, 1, 3, 3)


def test_metadata_validates_stage_and_final_stage() -> None:
    request = LayeredPrefillRequestData("request", 0, 16)
    assert not LayeredPrefillMetadata(0, 2, (request,)).is_final_stage
    assert LayeredPrefillMetadata(1, 2, (request,)).is_final_stage
    with pytest.raises(ValueError, match="stage"):
        LayeredPrefillMetadata(2, 2, (request,))


def test_supported_models_are_explicit() -> None:
    assert "Qwen3ForCausalLM" in LAYERED_PREFILL_MODEL_ADAPTERS
    assert LAYERED_PREFILL_MODEL_ADAPTERS["Qwen3MoeForCausalLM"].is_moe
