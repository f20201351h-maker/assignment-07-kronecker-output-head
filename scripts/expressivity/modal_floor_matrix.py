"""Run the contextual floor measurements on a Modal GPU (reference-stability matrix).

Same code as the CPU runs (`floor_contextual.py --device cuda`), same data (volume `kq5-data`: token arrays at the
historical path, checkpoints and per-position NLL arrays under /runs). Outputs are written to the volume under
/expressivity_gpu and returned to the local entrypoint, which stores them in results/expressivity/gpu/.

    export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8 PYTHONUTF8=1
    modal run scripts/expressivity/modal_floor_matrix.py::stability
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
try:                                   # locally: <repo>/scripts/expressivity -> <repo>
    ROOT = HERE.parents[1]
except IndexError:                     # inside the container the entry file is mounted at /root
    ROOT = Path("/root/proj")
app = modal.App("kq5-expressivity-matrix")
vol = modal.Volume.from_name("kq5-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("torch==2.10.0", "numpy==2.2.6")
if (ROOT / "src").exists():
    image = (image.add_local_dir(str(ROOT / "src"), remote_path="/root/proj/src")
             .add_local_dir(str(HERE), remote_path="/root/proj/scripts/expressivity"))
GPU = os.environ.get("MODAL_GPU", "L4")
TOKENS = "/data/C:/Program Files/Git/tokens"      # historical upload path (hash-verified by TokenData in the past)


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, timeout=2 * 3600)
def run_cmd(job: dict) -> dict:
    """job = {"name": str, "args": [cli args for floor_contextual.py]}; returns the produced files as bytes."""
    import subprocess
    import sys
    import time
    proj = Path("/root/proj")
    (proj / "data").mkdir(exist_ok=True)
    if not (proj / "data/tokens").exists():
        os.symlink(TOKENS, proj / "data/tokens")
    for x in ("m3", "p2", "p3", "p4"):
        d = proj / f"experiments/modal-{x}"
        d.mkdir(parents=True, exist_ok=True)
        if not (d / "runs").exists():
            os.symlink("/data/runs", d / "runs")
    ev = proj / "results/expressivity"
    ev.mkdir(parents=True, exist_ok=True)
    if not (proj / "checkpoints").exists():
        os.symlink("/data/runs", proj / "checkpoints")
    out_json = job["name"] + ".json"
    cmd = [sys.executable, str(proj / "scripts/expressivity/floor_contextual.py"), *job["args"], "--device", "cuda", "--out", out_json]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(proj), capture_output=True, text=True)
    log = f"$ {' '.join(cmd)}\n\n{proc.stdout}\n{proc.stderr}\n[exit {proc.returncode}, {time.time() - t0:.0f}s]\n"
    files = {job["name"] + ".log": log.encode()}
    for ext in (".json", ".npz"):
        f = ev / (job["name"] + ext)
        if f.exists():
            files[job["name"] + ext] = f.read_bytes()
    dst = Path("/data/expressivity_gpu")
    dst.mkdir(exist_ok=True)
    for k, v in files.items():
        (dst / k).write_bytes(v)
    vol.commit()
    import torch
    files["_device"] = torch.cuda.get_device_name(0).encode() if torch.cuda.is_available() else b"cpu"
    return files


def jobs() -> list[dict]:
    common = ["--windows", "32", "--per-window", "128", "--iters", "200", "--chunk", "512", "--enrich-contexts", "4096",
              "--configs", "kas", "kas+suffix", "kas+trigram_free", "kas+private4000", "kas+hash8192"]
    out = []
    for seed in (1, 2):
        ck = f"checkpoints/p4-densep-s{seed}/model.pt"
        for split in ("dev", "test"):
            out.append({"name": f"floor_gpu_s{seed}_{split}", "args": [*common, "--ckpt", ck, "--split", split]})
    return out


@app.local_entrypoint()
def stability():
    dst = ROOT / "results/expressivity/gpu"
    dst.mkdir(parents=True, exist_ok=True)
    for files in run_cmd.map(jobs()):
        dev = files.pop("_device", b"?").decode()
        for k, v in files.items():
            (dst / k).write_bytes(v)
        print("received", sorted(files), "device", dev, flush=True)
