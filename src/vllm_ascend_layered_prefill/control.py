# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-independent control data migrated from legacy Ascend PR #272."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal


@dataclass(frozen=True)
class LayeredPrefillModelAdapter:
    layer_call_order: Literal["positions_first", "hidden_first"]
    embedding_method: str = "embed_input_ids"
    is_moe: bool = False


LAYERED_PREFILL_MODEL_ADAPTERS = MappingProxyType(
    {
        "Qwen3ForCausalLM": LayeredPrefillModelAdapter("positions_first"),
        "Qwen3MoeForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
        "GptOssForCausalLM": LayeredPrefillModelAdapter("hidden_first", is_moe=True),
        "MixtralForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
        "Glm4MoeForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
        "Ernie4_5_MoeForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
        "DeepseekForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
        "DeepseekV2ForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
        "DeepseekV3ForCausalLM": LayeredPrefillModelAdapter(
            "positions_first", is_moe=True
        ),
    }
)

_MOE_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def get_moe_layer_cursors(
    moe_layer_names: Sequence[str], start_layer: int, end_layer: int
) -> tuple[int, ...]:
    if start_layer < 0 or end_layer < start_layer:
        raise ValueError(f"invalid decoder layer range [{start_layer}, {end_layer})")
    layer_indices: list[int] = []
    for name in moe_layer_names:
        match = _MOE_LAYER_PATTERN.search(name)
        if match is None:
            raise ValueError(f"cannot map MoE module to transformer layer: {name!r}")
        layer_index = int(match.group(1))
        if not start_layer <= layer_index < end_layer:
            raise ValueError(f"MoE module is outside decoder layer range: {name!r}")
        layer_indices.append(layer_index)
    if layer_indices != sorted(layer_indices):
        raise ValueError("MoE modules are not in transformer execution order")
    cursors: list[int] = []
    moe_index = 0
    for boundary in range(start_layer, end_layer + 1):
        while moe_index < len(layer_indices) and layer_indices[moe_index] < boundary:
            moe_index += 1
        cursors.append(moe_index)
    return tuple(cursors)


@dataclass(frozen=True)
class LayeredPrefillRequestData:
    req_id: str
    start_token: int
    num_tokens: int

    def __post_init__(self) -> None:
        if not self.req_id or self.start_token < 0 or self.num_tokens <= 0:
            raise ValueError("invalid layered-prefill request data")


@dataclass(frozen=True)
class LayeredPrefillMetadata:
    stage: int
    num_stages: int
    requests: tuple[LayeredPrefillRequestData, ...]

    def __post_init__(self) -> None:
        if self.num_stages < 2 or not 0 <= self.stage < self.num_stages:
            raise ValueError("invalid layered-prefill stage")

    @property
    def is_final_stage(self) -> bool:
        return self.stage + 1 == self.num_stages


def get_layer_stage_range(
    num_layers: int, num_stages: int, stage: int
) -> tuple[int, int]:
    if not 0 <= stage < num_stages:
        raise ValueError(f"invalid stage {stage} for {num_stages} stages")
    if num_stages > num_layers:
        raise ValueError(f"{num_stages} stages but only {num_layers} layers")
    base, remainder = divmod(num_layers, num_stages)
    start = stage * base + min(stage, remainder)
    return start, start + base + (1 if stage < remainder else 0)


__all__ = [
    "LAYERED_PREFILL_MODEL_ADAPTERS",
    "LayeredPrefillMetadata",
    "LayeredPrefillModelAdapter",
    "LayeredPrefillRequestData",
    "get_layer_stage_range",
    "get_moe_layer_cursors",
]
