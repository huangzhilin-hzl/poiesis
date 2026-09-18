"""Run a repository Python script on a Modal B300.

uvx --from 'modal[api-proxy-support]' modal run modal_run.py -- \
    experiments/ops/cutedsl/tma_v0.py --M 512 --N 128
"""

from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/workspace/poiesis"
IGNORE = modal.FilePatternMatcher.from_file(PROJECT_ROOT / ".modalignore")

app = modal.App("poiesis-run")
image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([])
    .pip_install("torch==2.12.0", index_url="https://download.pytorch.org/whl/cu132")
    .pip_install("nvidia-cutlass-dsl[cu13]==4.7.1", "numpy==2.2.6")
    .env({"PYTHONUNBUFFERED": "1"})
    .workdir(REMOTE_ROOT)
    .add_local_dir(PROJECT_ROOT, REMOTE_ROOT, copy=False, ignore=IGNORE)
)


def resolve_script(script: str) -> str:
    """Return a repository-relative path, rejecting missing or excluded scripts."""
    path = (PROJECT_ROOT / script).resolve()
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ValueError(f"Script must be inside {PROJECT_ROOT}: {script}") from None
    if path.suffix != ".py" or not path.is_file():
        raise ValueError(f"Python script does not exist: {path}")
    if IGNORE(relative):
        raise ValueError(f"Script is excluded by .modalignore: {relative}")
    return relative.as_posix()


@app.function(image=image, gpu="B300", cpu=4, memory=8192, timeout=600)
def run(script: str, script_args: list[str]):
    import os
    import shlex
    import subprocess
    import sys
    from importlib.metadata import version

    import torch

    command = [sys.executable, "-u", f"{REMOTE_ROOT}/{script}", *script_args]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (REMOTE_ROOT, env.get("PYTHONPATH")) if part
    )
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
    print(f"CuTe DSL: {version('nvidia-cutlass-dsl')}", flush=True)
    print(f"Working directory: {REMOTE_ROOT}", flush=True)
    print(f"Command: {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=REMOTE_ROOT, env=env, check=True)


def execute_script(script: str, script_args: list[str]):
    """Shared local dispatch for the generic and legacy entrypoints."""
    run.remote(resolve_script(script), script_args)


@app.local_entrypoint()
def main(*script_args):
    """Pass SCRIPT.py followed by its arguments; use -- to forward --help too."""
    if not script_args:
        raise SystemExit(
            "Usage: modal run modal_run.py -- <repository/script.py> [script arguments]"
        )
    execute_script(script_args[0], list(script_args[1:]))
