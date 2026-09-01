# Layered prefill host contract proposal

The extracted control data, stage partitioning, and MoE cursor logic are
host-independent. Runtime activation is a vLLM Ascend extension and requires:

1. `vllm.scheduler.layered-prefill.v1`: transport stage metadata without adding
   extension fields directly to core `SchedulerOutput`.
2. `vllm.model.partial-forward.v1`: execute a contiguous decoder layer range and
   return resumable intermediate tensors.
3. `vllm.worker.request-state.v1`: retain and release per-request intermediates
   with cancellation and failure cleanup.
4. `vllm.ascend.moe-cursor.v1`: set a validated MoE registry cursor before a
   partial forward.

The vLLM Ascend worker remains lifecycle owner. The extension must not replace
the worker class or patch a model instance's `forward` method. Upstream-derived
code must retain its MIT notice alongside Huawei and Apache notices.
