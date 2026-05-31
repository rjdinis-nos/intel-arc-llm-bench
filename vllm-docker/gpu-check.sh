#!/usr/bin/env bash
# Quick Intel XPU visibility check for the vLLM image.
# Runs `sycl-ls` and torch's XPU probe inside the built image WITHOUT booting
# the full vLLM server — handy after an Intel driver / WSL update to confirm the
# GPU is actually exposed before debugging a slow model load.
#
# Usage:
#   ./gpu-check.sh            # auto-detect WSL2 (/dev/dxg) vs native (/dev/dri)
#   VLLM_IMAGE=... ./gpu-check.sh
set -euo pipefail

VLLM_IMAGE="${VLLM_IMAGE:-vllm-xpu:v0.22.0}"

# The image's own oneAPI LD_LIBRARY_PATH. Compose env vars REPLACE rather than
# append, so any override below must carry the full list or torch fails to find
# libccl.so.1 (see docker-compose.wsl.yml for the same workaround).
ONEAPI_LD="/opt/intel/oneapi/tcm/1.4/lib:/opt/intel/oneapi/umf/1.0/lib:/opt/intel/oneapi/tbb/2022.3/env/../lib/intel64/gcc4.8:/opt/intel/oneapi/pti/0.16/lib:/opt/intel/oneapi/mpi/2021.17/opt/mpi/libfabric/lib:/opt/intel/oneapi/mpi/2021.17/lib:/opt/intel/oneapi/mkl/2025.3/lib:/opt/intel/oneapi/dnnl/2025.3/lib:/opt/intel/oneapi/debugger/2025.3/opt/debugger/lib:/opt/intel/oneapi/compiler/2025.3/opt/compiler/lib:/opt/intel/oneapi/compiler/2025.3/lib:/opt/intel/oneapi/ccl/2021.17/lib/:/usr/local/lib/"

DOCKER_ARGS=()
if grep -qi microsoft /proc/version 2>/dev/null; then
	echo "Host: WSL2 (Intel GPU via /dev/dxg)"
	if [ ! -e /dev/dxg ]; then
		echo "✗ /dev/dxg not found — WSL GPU passthrough is unavailable." >&2
		exit 1
	fi
	DOCKER_ARGS+=(--device /dev/dxg
		-v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro
		-v /usr/lib/wsl/drivers:/usr/lib/wsl/drivers:ro
		-e "LD_LIBRARY_PATH=/usr/lib/wsl/lib:${ONEAPI_LD}")
else
	echo "Host: native Linux (Intel GPU via /dev/dri)"
	if [ ! -d /dev/dri ]; then
		echo "✗ /dev/dri not found — no native GPU render nodes." >&2
		exit 1
	fi
	DOCKER_ARGS+=(--device /dev/dri
		--group-add "${RENDER_GID:-render}"
		--group-add "${VIDEO_GID:-video}"
		-e "LD_LIBRARY_PATH=${ONEAPI_LD}")
fi

echo "Image: ${VLLM_IMAGE}"
echo "----------------------------------------------------------------"

docker run --rm "${DOCKER_ARGS[@]}" --entrypoint bash "${VLLM_IMAGE}" -c '
	echo "=== sycl-ls (look for a [level_zero:gpu] / [opencl:gpu] entry) ==="
	sycl-ls 2>&1 || true
	echo
	echo "=== torch XPU probe ==="
	python -c "import torch; ok=torch.xpu.is_available(); print(\"xpu available:\", ok); print(\"device count:\", torch.xpu.device_count()); [print(\"device 0:\", torch.xpu.get_device_name(0)) if ok else None]" 2>&1
'

echo "----------------------------------------------------------------"
echo "If torch reports 'xpu available: True', the GPU is visible — run: make start"
echo "If it shows only a CPU device, the host Intel driver isn't exposing"
echo "Level-Zero compute yet (update the Intel Arc driver, then 'wsl --update')."
