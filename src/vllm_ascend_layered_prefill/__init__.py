"""Layered-prefill control contracts and inert runtime metadata."""

from .control import (
    LAYERED_PREFILL_MODEL_ADAPTERS,
    LayeredPrefillMetadata,
    LayeredPrefillModelAdapter,
    LayeredPrefillRequestData,
    get_layer_stage_range,
    get_moe_layer_cursors,
)


class VllmAscendLayeredPrefillContractProposal:
    """Metadata-only proposal; this class performs no runtime activation."""


__all__ = [
    "LAYERED_PREFILL_MODEL_ADAPTERS",
    "LayeredPrefillMetadata",
    "LayeredPrefillModelAdapter",
    "LayeredPrefillRequestData",
    "VllmAscendLayeredPrefillContractProposal",
    "get_layer_stage_range",
    "get_moe_layer_cursors",
]
