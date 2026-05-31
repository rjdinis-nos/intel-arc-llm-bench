"""Runtime patch for Intel XPU under WSL2.

The WSL2 paravirtual Intel GPU driver does not implement the Level-Zero
"query free memory" call, so ``torch.xpu.mem_get_info()`` raises:

    RuntimeError: The device (...) doesn't support querying the available
    free memory.

vLLM calls this during startup (MemorySnapshot in the XPU worker) to size the
KV cache, which makes ``vllm serve`` crash with "Engine core initialization
failed" on WSL2 even though the GPU is otherwise fully functional.

This module is imported automatically by CPython at interpreter startup
(``sitecustomize`` is on ``PYTHONPATH``), including the spawned engine-core
subprocesses. It wraps ``mem_get_info`` so that, when the native query is
unsupported, it falls back to reporting system memory as free (correct for the
UMA/shared-memory iGPU) and the device's total memory from device properties.

On native Linux (or any device where the query works) the wrapper is a no-op:
the original implementation is used and the fallback never triggers.
"""

try:
    import torch

    _orig_mem_get_info = torch.xpu.mem_get_info

    def _safe_mem_get_info(device=None):
        try:
            return _orig_mem_get_info(device)
        except RuntimeError:
            import psutil

            props = torch.xpu.get_device_properties(device if device is not None else 0)
            total = int(props.total_memory)
            # UMA iGPU shares system RAM; report available system memory as free,
            # capped at the device's reported total.
            free = min(int(psutil.virtual_memory().available), total)
            return (free, total)

    torch.xpu.mem_get_info = _safe_mem_get_info
    torch.xpu.memory.mem_get_info = _safe_mem_get_info
except Exception:
    # Never block interpreter startup if torch/xpu isn't importable.
    pass
