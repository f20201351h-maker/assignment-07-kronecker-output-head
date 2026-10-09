"""Kaggle entrypoint template. stage_kernel.py embeds the kq5 source zip and the job
spec into a copy of this file; that copy is what Kaggle runs.

Two independent single-GPU workers (one per T4), no DDP. Each worker runs its task
list sequentially, writing everything under /kaggle/working/runs/<run_id>/.
"""

import base64
import io
import json
import os
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

SRC_ZIP_B64 = "__SRC_ZIP_B64__"
SPEC = json.loads(r'''__SPEC_JSON__''')
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
T0 = time.time()


def unpack_source() -> Path:
    d = WORK / "src"
    if not (d / "kq5").exists():
        zipfile.ZipFile(io.BytesIO(base64.b64decode(SRC_ZIP_B64))).extractall(d)
    return d


def find_data() -> Path:
    hits = [p.parent for p in INPUT.rglob("manifest.json")
            if (p.parent / "vocab.json").exists() and (p.parent / "train.npy").exists()]
    if len(hits) != 1:
        mounts = [str(p) for p in INPUT.rglob("*")][:50] if INPUT.exists() else "no /kaggle/input"
        raise RuntimeError(f"expected one token directory, found {hits}; mounts={mounts}")
    return hits[0]


def device_report() -> dict:
    import torch
    rep = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
           "device_count": torch.cuda.device_count(), "elapsed_s": time.time() - T0}
    if torch.cuda.is_available():
        rep["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        rep["cuda"] = torch.version.cuda
    try:
        rep["nvidia_smi"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True,
                                           timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        rep["nvidia_smi"] = f"unavailable: {e}"
    return rep


def run_task(task: dict, data_dir: Path) -> dict:
    sys.path.insert(0, str(unpack_source()))
    kind = task["type"]
    if kind == "train":
        from kq5.train import RunConfig, train
        cfg = dict(task["config"])
        cfg["out_dir"] = str(WORK / "runs" / cfg["run_id"])
        cfg["data_dir"] = str(data_dir)
        return train(RunConfig(**cfg))
    if kind == "python":
        mod = __import__(task["module"], fromlist=["main"])
        return mod.main(data_dir=data_dir, work=WORK, **task.get("kwargs", {}))
    raise ValueError(kind)


def worker(gpu: int) -> None:
    data_dir = find_data()
    tasks = SPEC["workers"][str(gpu)]
    status_path = WORK / f"worker{gpu}_status.jsonl"
    for task in tasks:
        name = task.get("name") or task.get("config", {}).get("run_id")
        remaining = SPEC["deadline_s"] - (time.time() - T0_GLOBAL)
        if remaining < task.get("est_s", 0):
            rec = {"task": name, "status": "skipped_deadline", "remaining_s": remaining}
        else:
            t = time.time()
            try:
                if task["type"] == "train":
                    task.setdefault("config", {})["time_limit_s"] = min(
                        task["config"].get("time_limit_s", 1e9), remaining - 600)
                res = run_task(task, data_dir)
                rec = {"task": name, "status": "ok", "wall_s": time.time() - t,
                       "result": res if isinstance(res, dict) else str(res)}
            except Exception as e:  # noqa: BLE001
                rec = {"task": name, "status": "error", "error": repr(e),
                       "traceback": traceback.format_exc()[-4000:], "wall_s": time.time() - t}
        with open(status_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "result"})[:2000], flush=True)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]))
        return
    WORK.mkdir(parents=True, exist_ok=True)
    unpack_source()
    rep = device_report()
    rep["spec_name"] = SPEC["name"]
    (WORK / "device.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2), flush=True)
    need = SPEC.get("require_gpus", 2)
    if rep["device_count"] < need or not all("T4" in d for d in rep.get("devices", [])):
        raise RuntimeError(f"accelerator check failed: need {need} T4, got {rep.get('devices')}")
    data_dir = find_data()
    print(f"data: {data_dir}", flush=True)
    procs = []
    for gpu in sorted(SPEC["workers"], key=int):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["KQ5_T0"] = str(T0)
        log = open(WORK / f"worker{gpu}.log", "w", encoding="utf-8")
        p = subprocess.Popen([sys.executable, __file__, "worker", gpu], env=env, stdout=log,
                             stderr=subprocess.STDOUT)
        procs.append((gpu, p, log))
        print(f"started worker {gpu} pid {p.pid}", flush=True)
    mon = WORK / "gpu_monitor.jsonl"
    while any(p.poll() is None for _, p, _ in procs):
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,"
                                "temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=30).stdout
        except Exception as e:  # noqa: BLE001
            q = f"error {e}"
        with open(mon, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time() - T0, "smi": q}) + "\n")
        time.sleep(30)
    codes = {}
    for gpu, p, log in procs:
        codes[gpu] = p.wait()
        log.close()
    summary = {"name": SPEC["name"], "exit_codes": codes, "wall_s": time.time() - T0}
    (WORK / "job_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    for gpu in codes:
        print(f"--- worker {gpu} tail ---\n" + (WORK / f"worker{gpu}.log").read_text()[-3000:], flush=True)


T0_GLOBAL = float(os.environ.get("KQ5_T0", T0))

if __name__ == "__main__":
    main()
