"""Run sm103_mla.py on a single Modal B300 (SM103).

From the repository root:
    uvx --from 'modal[api-proxy-support]' modal setup
    uvx --from 'modal[api-proxy-support]' modal run experiments/model/glm53/modal_sm103_mla.py

Only this runner and the benchmark are uploaded; no model weights are needed.
The local CLI extra supports HTTP/SOCKS proxies configured on the host.
Append the benchmark's flags, e.g. --mode profile --chunk 3
--trace-path /tmp/mla_profile.json. Profile traces are downloaded to the local
--trace-path; the remote process uses a temporary output directory.
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
def run(script_args: list[str], capture_trace: bool = False) -> bytes | None:
    import subprocess
    import sys
    from importlib.metadata import version
    from tempfile import TemporaryDirectory

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

    # A subprocess gives argparse only the benchmark's arguments, without any
    # Modal worker arguments, and keeps profiler state isolated between calls.
    with TemporaryDirectory(prefix="mla-profile-") as output_dir:
        remote_trace = Path(output_dir) / "mla_profile.json"
        command = [sys.executable, "-u", REMOTE_SCRIPT, *script_args]
        if capture_trace:
            command.extend(["--trace-path", str(remote_trace)])
        subprocess.run(command, check=True)
        return remote_trace.read_bytes() if capture_trace else None


@app.local_entrypoint()
def main(*script_args):
    """Forward benchmark flags and save --trace-path on the local machine."""
    import argparse
    import time

    # Interpret only the options needed to retrieve the output. The benchmark's
    # own argparse parser handles every other flag, including future additions.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--mode", choices=("bench", "profile"), default="bench")
    parser.add_argument("--trace-path", type=Path)
    options, remaining = parser.parse_known_args(script_args)
    capture_trace = options.mode == "profile"
    local_trace = options.trace_path or Path(f"sm103_mla_profile_{time.time_ns()}.json")

    trace = run.remote(["--mode", options.mode, *remaining], capture_trace=capture_trace)
    if capture_trace:
        if trace is None:
            raise RuntimeError("The remote profile run returned no trace")
        local_trace = local_trace.expanduser().resolve()
        local_trace.parent.mkdir(parents=True, exist_ok=True)
        local_trace.write_bytes(trace)
        print(f"[PROFILE] Trace downloaded to local file: {local_trace}")
