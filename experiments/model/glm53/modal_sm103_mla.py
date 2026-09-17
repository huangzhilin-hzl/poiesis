"""Run sm103_mla.py on a single Modal B300 (SM103).

From the repository root:
    uvx --from 'modal[api-proxy-support]' modal setup
    uvx --from 'modal[api-proxy-support]' modal run experiments/model/glm53/modal_sm103_mla.py

Only this runner and the benchmark are uploaded; no model weights are needed.
The local CLI extra supports HTTP/SOCKS proxies configured on the host.
"""

from pathlib import Path

import modal

FLASHINFER_VERSION = "0.6.18.post1"
REMOTE_SCRIPT = "/root/sm103_mla.py"

app = modal.App("poiesis-sm103-mla")

# B300 requires CUDA >= 13.1 on Modal. Keep the toolkit, PyTorch wheel,
# and CUPTI in the same CUDA 13.2 family.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.2.1-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install("torch==2.12.0", index_url="https://download.pytorch.org/whl/cu132")
    .pip_install(
        f"flashinfer-python[cu13]=={FLASHINFER_VERSION}",
        "numpy==2.2.6",
        "cupti-python==13.2.0",
    )
    .pip_install(
        f"flashinfer-cubin=={FLASHINFER_VERSION}",
        index_url="https://flashinfer.ai/whl",
    )
    .env({"PYTHONUNBUFFERED": "1", "MAX_JOBS": "4"})
    .add_local_file(Path(__file__).with_name("sm103_mla.py"), REMOTE_SCRIPT)
)


@app.function(image=image, gpu="B300", cpu=4, memory=8192, timeout=1800)
def run():
    import runpy
    import subprocess
    from importlib.metadata import version

    import flashinfer
    import torch
    from cupti import cupti  # Ensure the requested timing backend is installed.

    subprocess.run(["nvidia-smi"], check=True)
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}, compute capability: {capability}")
    print(f"PyTorch: {torch.__version__}, CUDA runtime: {torch.version.cuda}")
    print(f"FlashInfer: {flashinfer.__version__}")
    print(f"CUPTI ({cupti.__name__}): {version('cupti-python')}")
    if capability != (10, 3):
        raise RuntimeError(f"This SM103 benchmark expects B300; got {capability}")

    runpy.run_path(REMOTE_SCRIPT, run_name="__main__")


@app.local_entrypoint()
def main():
    run.remote()
