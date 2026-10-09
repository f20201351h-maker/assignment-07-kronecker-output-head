"""Modal runner for every canonical and later run (protocol 05 onwards; the pilot ran on Kaggle T4s).
Same `kq5` code, same token arrays (Modal Volume `kq5-data`, sha256-verified by TokenData.load), same RunConfig.

    modal run --detach scripts/modal_app.py::canonical_detached --gpu RTX-PRO-6000   # Phase 1 (m3), 14 runs
    modal run --detach scripts/modal_app.py::phase2_detached --gpu RTX-PRO-6000      # Phase 2 (p2)
    modal run --detach scripts/modal_app.py::confirm_detached --gpu RTX-PRO-6000     # protocol 07 (p3)
    modal run --detach scripts/modal_app.py::prior_detached --gpu RTX-PRO-6000       # protocol 08 (p4)
    modal volume get kq5-data /export/<run_id>.tar.gz experiments/<set>/export/
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("kq5-output-heads")
vol = modal.Volume.from_name("kq5-data", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.10.0", "numpy==2.2.6")
    .add_local_dir(str(ROOT / "src"), remote_path="/root/src")
)
GPU = "H100"
# Git Bash path conversion placed the uploaded token directory at this path inside the volume
# (`modal volume put kq5-data data/tokens /tokens` was rewritten). TokenData.load verifies every
# file's sha256 against manifest.json, so the location does not affect what is read.
DATA_DIR = "/data/C:/Program Files/Git/tokens"


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, timeout=3 * 3600)
def train_run(cfg: dict) -> dict:
    import sys
    import torch
    sys.path.insert(0, "/root/src")
    from kq5.train import RunConfig, train
    cfg = dict(cfg)
    cfg["data_dir"] = DATA_DIR
    cfg["out_dir"] = f"/data/runs/{cfg['run_id']}"
    summary = train(RunConfig(**cfg))
    summary["device"] = torch.cuda.get_device_name(0)
    Path(cfg["out_dir"], "device.json").write_text(json.dumps({"device": summary["device"],
                                                                "torch": torch.__version__}), encoding="utf-8")
    vol.commit()
    return summary


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, timeout=2 * 3600)
def analyse_run(run_id: str, splits: list[str]) -> dict:
    import sys
    sys.path.insert(0, "/root/src")
    from kq5.analysis import analyze_run
    out = {}
    for sp in splits:
        out[sp] = analyze_run(Path(f"/data/runs/{run_id}"), Path(DATA_DIR), sp)
    vol.commit()
    return out


def r2_style(run_id: str, head: str, seed: int, **extra) -> dict:
    """The pilot (R2) configuration: 8L x 512, 20,971,520 tokens, head lr x0.1, dev evaluation."""
    cfg = dict(run_id=run_id, head=head, seed=seed, n_layer=8, n_head=8, d_model=512, seq_len=1024,
               total_tokens=20 * 1048576, warmup_tokens=1048576, batch_seqs=64, micro_seqs=8,
               eval_at_tokens=[4 * 1048576, 8 * 1048576, 12 * 1048576, 16 * 1048576],
               eval_windows_mid=None, final_dev_windows=None, save_checkpoint=True, log_every=10,
               head_lr_mult=0.1)
    cfg.update(extra)
    return cfg


V3 = {"emb": 64, "channels": 256, "layers": 3, "kernel": 3, "hidden": 512}


@app.local_entrypoint()
def check_init():
    """Exploratory: does the u_zero initialisation let the stored and generated corrections learn
    at the pilot scale? Compare with the Kaggle pilot's KAS-P (1.9343 / 1.9371 dev bpb)."""
    cfgs = [r2_style("m2-kasu0-s1", "kasu", 1, corr_init="u_zero"),
            r2_style("m2-kasg0-v3-s1", "kasg", 1, corr_init="u_zero", generator=V3, gen_lr_mult=1.0),
            r2_style("m2-kasp-s1", "kasp", 1)]
    results = list(train_run.map(cfgs))
    for c, r in zip(cfgs, results):
        print(json.dumps({"run": c["run_id"], "device": r.get("device"), "status": r["status"],
                          "dev_bpb": r["final"]["dev"]["bpb"], "wall_s": round(r["wall_s"])}))


@app.function(image=image, gpu=GPU, timeout=600)
def ping() -> dict:
    import torch
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "available": torch.cuda.is_available()}


@app.local_entrypoint()
def check_gpu():
    print(json.dumps(ping.remote()))


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, timeout=1200)
def probe(cfg: dict) -> dict:
    """Throughput probe: a few dozen training steps, no checkpoint; median tok/s after step 10."""
    import sys
    import torch
    sys.path.insert(0, "/root/src")
    from kq5.train import RunConfig, train
    cfg = dict(cfg)
    cfg["data_dir"] = DATA_DIR
    cfg["out_dir"] = f"/tmp/probe/{cfg['run_id']}"
    train(RunConfig(**cfg))
    lines = [json.loads(l) for l in open(f"/tmp/probe/{cfg['run_id']}/metrics.jsonl")]
    tps = sorted(l["tok_per_s"] for l in lines if l.get("tok_per_s") and l["step"] > 10)
    return {"device": torch.cuda.get_device_name(0), "tok_per_s": tps[len(tps) // 2] if tps else None,
            "peak_mem_gb": max((l.get("peak_mem_gb", 0) for l in lines), default=None)}


@app.local_entrypoint()
def probe_gpus():
    gpus = ["H200", "RTX-PRO-6000", "A100-40GB", "A100-80GB", "L40S", "A10", "L4"]
    base = dict(n_layer=8, n_head=8, d_model=512, seq_len=1024, total_tokens=100_007_936,
                stop_tokens=32 * 65536, batch_seqs=64, micro_seqs=8, eval_at_tokens=[], final_dev_windows=4,
                save_checkpoint=False, log_every=4, head_lr_mult=0.1)
    jobs = []
    for g in gpus:
        for head, extra in (("kasp", {}), ("kasg", {"generator": V3, "corr_init": "u_zero"})):
            cfg = dict(base, run_id=f"probe-{g}-{head}", head=head, **extra)
            try:
                jobs.append((g, head, probe.with_options(gpu=g).spawn(cfg)))
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"gpu": g, "head": head, "error": repr(e)[:200]}))
    for g, head, fc in jobs:
        try:
            r = fc.get()
            print(json.dumps({"gpu": g, "head": head, **r}))
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"gpu": g, "head": head, "error": repr(e)[:300]}))


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, timeout=4 * 3600)
def canonical_run(cfg: dict, retok_rules: list) -> dict:
    """Primary canonical run (Protocol 05): train, then dev and test analysis, then a tarball of the
    run directory without the checkpoint for download. Commits the volume after each stage."""
    import subprocess
    import sys
    import torch
    sys.path.insert(0, "/root/src")
    from kq5.analysis import analyze_run
    from kq5.train import RunConfig, train
    cfg = dict(cfg)
    rid = cfg["run_id"]
    cfg["data_dir"] = DATA_DIR
    cfg["out_dir"] = f"/data/runs/{rid}"
    summary = train(RunConfig(**cfg))
    dev = torch.cuda.get_device_name(0)
    Path(cfg["out_dir"], "device.json").write_text(json.dumps({"device": dev, "torch": torch.__version__}),
                                                   encoding="utf-8")
    vol.commit()
    ana = {}
    for sp in ("dev", "test"):
        try:
            ana[sp] = analyze_run(Path(cfg["out_dir"]), Path(DATA_DIR), sp, retok_rules=tuple(retok_rules))
        except Exception as e:  # noqa: BLE001
            import traceback
            ana[sp] = {"error": repr(e), "traceback": traceback.format_exc()[-2000:]}
    Path("/data/export").mkdir(exist_ok=True)
    subprocess.run(["tar", "czf", f"/data/export/{rid}.tar.gz", "--exclude=model.pt", "-C", "/data/runs", rid],
                   check=True)
    vol.commit()
    return {"run_id": rid, "device": dev, "status": summary["status"], "wall_s": summary["wall_s"],
            "dev_bpb": summary["final"]["dev"]["bpb"], "test_bpb": summary["final"].get("test", {}).get("bpb"),
            "analysis": {k: ("ok" if "error" not in v else v["error"]) for k, v in ana.items()}}


@app.function(image=image, gpu=GPU, volumes={"/data": vol}, timeout=3 * 3600)
def inference_task(module: str, kwargs: dict) -> dict:
    """Growth (kq5.growth) or head cost (kq5.headcost) on checkpoints already in the volume."""
    import sys
    sys.path.insert(0, "/root/src")
    mod = __import__(module, fromlist=["main"])
    res = mod.main(data_dir=Path(DATA_DIR), work=Path("/data"), **kwargs)
    vol.commit()
    return {"module": module, "result": res}


def canonical_specs():
    """The exact train configs and analysis rules of Kaggle kernels X and G, ids r3- -> m3-."""
    runs, growth = [], []
    for name in ("r3_canonical_x", "r3_canonical_g"):
        spec = json.loads((ROOT / "configs" / "jobs" / f"{name}.json").read_text(encoding="utf-8"))
        for w in spec["workers"].values():
            for t in w:
                if t["type"] == "train":
                    c = dict(t["config"]); c["run_id"] = c["run_id"].replace("r3-", "m3-", 1)
                    runs.append(c)
                elif t["type"] == "python" and t["module"] == "kq5.analysis":
                    rules = t["kwargs"]["retok_rules"]
                elif t["type"] == "python" and t["module"] == "kq5.growth":
                    growth.append([r.replace("r3-", "m3-", 1) for r in t["kwargs"]["runs"]])
    return runs, rules, growth


@app.local_entrypoint()
def canonical_all(gpu: str = "H100"):
    runs, rules, growth = canonical_specs()
    print(json.dumps({"n_runs": len(runs), "ids": [r["run_id"] for r in runs], "gpu": gpu}))
    calls = [canonical_run.with_options(gpu=gpu).spawn(c, rules) for c in runs]
    results = []
    for fc in calls:
        try:
            results.append(fc.get())
        except Exception as e:  # noqa: BLE001
            results.append({"error": repr(e)[:500]})
        print(json.dumps(results[-1]), flush=True)
    post = [inference_task.with_options(gpu=gpu).spawn("kq5.growth", {"runs": g}) for g in growth]
    post.append(inference_task.with_options(gpu=gpu).spawn("kq5.headcost", {"d_model": 512, "BT": 8192,
                                                                          "chunk": 32768}))
    for fc in post:
        try:
            print(json.dumps(fc.get(), default=str)[:600], flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"error": repr(e)[:500]}), flush=True)


@app.function(image=image, volumes={"/data": vol}, timeout=8 * 3600)
def orchestrate(gpu: str, runs: list, rules: list, growth: list, tag: str = "canonical",
                headcost: bool = True) -> dict:
    """Runs on Modal (CPU) so the canonical set survives the local client disconnecting: spawns
    every canonical run, waits, then growth and head cost; writes a summary to the volume.
    tag names the progress/summary files (Phase 1: canonical_*; Phase 2: phase2_*)."""
    calls = [canonical_run.with_options(gpu=gpu).spawn(c, rules) for c in runs]
    results = []
    for fc in calls:
        try:
            results.append(fc.get())
        except Exception as e:  # noqa: BLE001
            results.append({"error": repr(e)[:800]})
        Path(f"/data/{tag}_progress.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
        vol.commit()
    post = [inference_task.with_options(gpu=gpu).spawn("kq5.growth", {"runs": g}) for g in growth]
    if headcost:
        post.append(inference_task.with_options(gpu=gpu).spawn("kq5.headcost", {"d_model": 512, "BT": 8192,
                                                                              "chunk": 32768}))
    post_res = []
    for fc in post:
        try:
            post_res.append(fc.get())
        except Exception as e:  # noqa: BLE001
            post_res.append({"error": repr(e)[:800]})
    out = {"gpu": gpu, "runs": results, "post": post_res}
    Path(f"/data/{tag}_summary.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    vol.commit()
    return out


# ---------------------------------------------------------------------------------------------
# Phase 2 (protocol 06, later-informed arms): the canonical KAS-U16 / KAS-G configurations with
# exactly one change each. Same data, order, budget, schedule, analysis rules and torch build.
# ---------------------------------------------------------------------------------------------
OCC = {"kind": "occmlp", "hidden": 92}      # reference report's generator form, parameter-matched to v3


def phase2_specs():
    runs, rules, _ = canonical_specs()
    base = {c["run_id"]: c for c in runs}

    def mk(src, rid, **change):
        c = dict(base[src])
        c["run_id"] = rid
        c.update(change)
        return c

    p2 = [mk("m3-kasu0-s1", "p2-kasu-drop25-s1", vocab_dropout=0.25),
          mk("m3-kasu0-s2", "p2-kasu-drop25-s2", vocab_dropout=0.25),
          mk("m3-kasg-s1", "p2-kasg-occ-s1", generator=OCC),
          mk("m3-kasg-s2", "p2-kasg-occ-s2", generator=OCC),
          mk("m3-kasu0-s1", "p2-kasu64-s1", rank=64),
          # amendment 06a: hide only the stored correction u_v (prior kept)
          mk("m3-kasu0-s1", "p2-kasu-dropu25-s1", vocab_dropout=0.25, vocab_dropout_target="u"),
          mk("m3-kasu0-s2", "p2-kasu-dropu25-s2", vocab_dropout=0.25, vocab_dropout_target="u"),
          # amendment 06a: second seed of rank 64 after seed 1 came out below Dense
          mk("m3-kasu0-s2", "p2-kasu64-s2", rank=64)]
    growth = [["p2-kasu-drop25-s1", "p2-kasg-occ-s1", "p2-kasu64-s1"], ["p2-kasu-dropu25-s1"]]
    return p2, rules, growth


@app.local_entrypoint()
def phase2_detached(gpu: str = "RTX-PRO-6000", only: str = ""):
    """modal run --detach scripts/modal_app.py::phase2_detached --gpu <GPU> [--only id1,id2]"""
    runs, rules, growth = phase2_specs()
    if only:
        keep = set(only.split(","))
        runs = [r for r in runs if r["run_id"] in keep]
        growth = [[r for r in g if r in keep] for g in growth]
        growth = [g for g in growth if g]
    fc = orchestrate.spawn(gpu, runs, rules, growth, "phase2", False)
    print(json.dumps({"spawned": fc.object_id, "n_runs": len(runs), "ids": [r["run_id"] for r in runs],
                      "gpu": gpu}))


def phase3_specs():
    """Protocol 07 confirmation pass: the frozen Dense (m3-dense-s1) and KAS-U64 (p2-kasu64-s1) configurations
    with fresh seeds 3 and 4 and nothing else changed (asserted)."""
    runs, rules, _ = canonical_specs()
    p2, _, _ = phase2_specs()
    base = {c["run_id"]: c for c in runs + p2}
    out = []
    for src, name in (("m3-dense-s1", "dense"), ("p2-kasu64-s1", "kasu64")):
        for seed in (3, 4):
            c = dict(base[src])
            c["run_id"] = f"p3-{name}-s{seed}"
            c["seed"] = seed
            changed = {k for k in set(c) | set(base[src]) if c.get(k) != base[src].get(k)}
            assert changed == {"run_id", "seed"}, (src, changed)
            out.append(c)
    return out, rules


@app.local_entrypoint()
def confirm_detached(gpu: str = "RTX-PRO-6000"):
    """modal run --detach scripts/modal_app.py::confirm_detached   (protocol 07; no growth task)"""
    runs, rules = phase3_specs()
    fc = orchestrate.spawn(gpu, runs, rules, [], "phase3", False)
    print(json.dumps({"spawned": fc.object_id, "n_runs": len(runs), "ids": [r["run_id"] for r in runs], "gpu": gpu}))


def phase4_specs():
    """Protocol 08 prior-matched Dense control: the four existing Dense configurations (m3-dense-s{1,2},
    p3-dense-s{3,4}) with `dense_bias=True` and nothing else changed (asserted). Same seed => W and the
    body are initialised bit-identically to the Dense run of that seed; the bias is the only difference."""
    runs, rules, _ = canonical_specs()
    p3, _ = phase3_specs()
    base = {c["run_id"]: c for c in runs + p3}
    out = []
    for src, seed in (("m3-dense-s1", 1), ("m3-dense-s2", 2), ("p3-dense-s3", 3), ("p3-dense-s4", 4)):
        c = dict(base[src])
        assert c["head"] == "dense" and c["seed"] == seed, (src, c["head"], c["seed"])
        c["run_id"] = f"p4-densep-s{seed}"
        c["dense_bias"] = True
        changed = {k for k in set(c) | set(base[src]) if c.get(k) != base[src].get(k)}
        assert changed == {"run_id", "dense_bias"}, (src, changed)
        out.append(c)
    return out, rules


@app.local_entrypoint()
def prior_detached(gpu: str = "RTX-PRO-6000"):
    """modal run --detach scripts/modal_app.py::prior_detached   (protocol 08; no growth task)"""
    runs, rules = phase4_specs()
    fc = orchestrate.spawn(gpu, runs, rules, [], "phase4", False)
    print(json.dumps({"spawned": fc.object_id, "n_runs": len(runs), "ids": [r["run_id"] for r in runs], "gpu": gpu}))


@app.local_entrypoint()
def probe_phase2(gpus: str = "B200,H100,RTX-PRO-6000"):
    """Short real-workload throughput probe of the Phase 2 arms (32 steps each) on candidate GPUs."""
    runs, _, _ = phase2_specs()
    base = {c["run_id"]: c for c in runs}
    jobs = []
    for g in gpus.split(","):
        for rid in ("p2-kasu-drop25-s1", "p2-kasg-occ-s1"):
            cfg = dict(base[rid], run_id=f"probe-{g}-{rid}", stop_tokens=32 * 65536, eval_at_tokens=[],
                       final_dev_windows=4, final_test=False, save_checkpoint=False, log_every=4)
            try:
                jobs.append((g, rid, probe.with_options(gpu=g).spawn(cfg)))
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"gpu": g, "run": rid, "error": repr(e)[:200]}))
    for g, rid, fc in jobs:
        try:
            print(json.dumps({"gpu": g, "run": rid, **fc.get()}), flush=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"gpu": g, "run": rid, "error": repr(e)[:300]}), flush=True)


@app.local_entrypoint()
def canonical_detached(gpu: str = "RTX-PRO-6000"):
    runs, rules, growth = canonical_specs()
    fc = orchestrate.spawn(gpu, runs, rules, growth)
    print(json.dumps({"spawned": fc.object_id, "n_runs": len(runs), "gpu": gpu}))
