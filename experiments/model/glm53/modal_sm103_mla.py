"""Run sm103_mla.py on a single Modal B300 (SM103).

From the repository root:
    uvx --from 'modal[api-proxy-support]' modal setup
    uvx --from 'modal[api-proxy-support]' modal run experiments/model/glm53/modal_sm103_mla.py

Only this runner and the benchmark are uploaded; no model weights are needed.
The local CLI extra supports HTTP/SOCKS proxies configured on the host.
Append the benchmark's flags, e.g. --mode profile --chunk 3. Profile traces stay
remote and are used to print kernel statistics. The default trace is deleted
with the temporary working directory after the run. Only --output-json is
downloaded. FlashMLA is compiled from the pinned upstream commit.
"""

from pathlib import Path

import modal

FLASHINFER_VERSION = "0.6.18.post1"
FLASHMLA_REVISION = "ba89a3466e9470ad08ab39738d4e7bb66989e1e7"
REMOTE_SCRIPT = "/root/sm103_mla.py"

app = modal.App("poiesis-sm103-mla")

# B300 requires CUDA >= 13.1 on Modal. Keep the toolkit, PyTorch wheel,
# and CUPTI in the same CUDA 13.2 family.
image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git", "build-essential", "ninja-build")
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
    .pip_install("setuptools", "wheel", "packaging", "ninja")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "MAX_JOBS": "2",
            "NVCC_THREADS": "2",
            "FLASH_MLA_DISABLE_SM90": "1",
            "FLASH_MLA_SOURCE_REVISION": FLASHMLA_REVISION,
        }
    )
    .run_commands(
        "git clone --no-checkout https://github.com/deepseek-ai/FlashMLA.git /opt/FlashMLA",
        f"git -C /opt/FlashMLA checkout {FLASHMLA_REVISION}",
        "git -C /opt/FlashMLA submodule update --init --recursive",
    )
    .run_commands(
        # Stream compiler output: pip's buffered failure summary can exceed
        # Modal's log-message limit and hide the actual compiler error.
        "python -m pip install -v --no-build-isolation --no-deps /opt/FlashMLA",
        # Modal's Python defaults to clang++, but this image provides GCC.
        env={"CC": "gcc", "CXX": "g++"},
    )
    .add_local_file(Path(__file__).with_name("sm103_mla.py"), REMOTE_SCRIPT)
)


@app.function(image=image, gpu="B300", cpu=4, memory=8192, timeout=1800)
def run(script_args: list[str], capture_json: bool = False) -> tuple[int, bytes | None]:
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
    print(f"FlashMLA: {version('flash_mla')}, source revision: {FLASHMLA_REVISION}")
    print(f"CUPTI ({cupti.__name__}): {version('cupti-python')}")
    if capability != (10, 3):
        raise RuntimeError(f"This SM103 benchmark expects B300; got {capability}")

    # A subprocess gives argparse only the benchmark's arguments, without any
    # Modal worker arguments, and keeps profiler state isolated between calls.
    with TemporaryDirectory(prefix="mla-run-") as output_dir:
        remote_json = Path(output_dir) / "mla_result.json"
        command = [sys.executable, "-u", REMOTE_SCRIPT, *script_args]
        if capture_json:
            command.extend(["--output-json", str(remote_json)])
        # The benchmark's default profile trace is created here, used for its
        # kernel summary, and cleaned up without being transferred to the client.
        completed = subprocess.run(command, check=False, cwd=output_dir)
        # A failed matrix may still have checkpoints from earlier workloads.
        report = (
            remote_json.read_bytes() if capture_json and remote_json.exists() else None
        )
        return completed.returncode, report


@app.local_entrypoint()
def main(*script_args):
    """Forward benchmark flags and optionally download the result JSON."""
    import argparse

    # Interpret only the options needed to retrieve the output. The benchmark's
    # own argparse parser handles every other flag, including future additions.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--output-json", type=Path)
    options, remaining = parser.parse_known_args(script_args)

    returncode, report = run.remote(
        remaining,
        capture_json=options.output_json is not None,
    )
    if options.output_json and report is not None:
        path = options.output_json.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(report)
        print(f"[RESULT] Report downloaded to local file: {path}")
    if returncode:
        raise SystemExit(returncode)
    if options.output_json and report is None:
        raise RuntimeError("The remote run returned no result JSON")
