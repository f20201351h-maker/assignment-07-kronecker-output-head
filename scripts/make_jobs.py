"""Write Kaggle job specs (configs/jobs/*.json). Each spec lists tasks per GPU worker."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "configs" / "jobs"

BACKBONES = {"8L512": dict(n_layer=8, n_head=8, d_model=512),
             "6L768": dict(n_layer=6, n_head=12, d_model=768)}


def train_task(run_id: str, head: str, est_s: float, **cfg) -> dict:
    c = {"run_id": run_id, "head": head}
    c.update(cfg)
    return {"type": "train", "name": run_id, "est_s": est_s, "config": c}


def r1() -> dict:
    gpu0 = []
    for bb, arch in BACKBONES.items():
        for head in ("dense", "kasp", "kasu", "kasg"):
            gpu0.append(train_task(f"r1-tput-{bb}-{head}", head, 300, **arch,
                                   total_tokens=100_000_000, stop_tokens=40 * 65536,
                                   batch_seqs=64, micro_seqs=8, eval_at_tokens=[],
                                   final_dev_windows=8, save_checkpoint=False, log_every=5))
    gpu1 = []
    for head, mult in (("kasp", 1.0), ("kasp", 0.1), ("kasp", 0.01), ("dense", 1.0)):
        gpu1.append(train_task(f"r1-lr-8L512-{head}-hm{mult}", head, 600, **BACKBONES["8L512"],
                               total_tokens=8 * 1048576, warmup_tokens=1048576, head_lr_mult=mult,
                               batch_seqs=64, micro_seqs=8, eval_at_tokens=[4 * 1048576, 8 * 1048576],
                               eval_windows_mid=256, final_dev_windows=256, save_checkpoint=False,
                               log_every=10))
    return {"name": "r1_smoke", "slug": "kq5-r1-smoke", "title": "kq5 r1 smoke",
            "deadline_s": 3 * 3600, "require_gpus": 2, "workers": {"0": gpu0, "1": gpu1}}


GEN_VARIANTS = {
    "v1": ({"emb": 64, "channels": 128, "layers": 2, "kernel": 3, "hidden": 256}, 1.0),
    "v2": ({"emb": 64, "channels": 128, "layers": 2, "kernel": 3, "hidden": 256}, 0.1),
    "v3": ({"emb": 64, "channels": 256, "layers": 3, "kernel": 3, "hidden": 512}, 1.0),
}
RETOK_RULES = ["mean", "inherit", "bytes", "count", "inherit_prior", "gen_count", "gen_inherit"]


def r2(backbone: str, head_lr_mult: float, est_s: float) -> dict:
    """Pilot at K2's 20M-token point: all arms, a seed pair for dense/kasp/kasu, three
    generator variants, the shuffled control; dev-split analysis after training."""
    arch = BACKBONES[backbone]
    common = dict(**arch, total_tokens=20 * 1048576, warmup_tokens=1048576, batch_seqs=64, micro_seqs=8,
                  eval_at_tokens=[4 * 1048576, 8 * 1048576, 12 * 1048576, 16 * 1048576],
                  eval_windows_mid=256, final_dev_windows=None, save_checkpoint=True, log_every=10,
                  head_lr_mult=head_lr_mult)

    def run(rid, head, seed, gv=None):
        extra = {}
        if gv is not None:
            g, glr = GEN_VARIANTS[gv]
            extra = {"generator": g, "gen_lr_mult": glr}
        cfg = dict(common, seed=seed, **extra)
        if head == "dense":
            cfg["head_lr_mult"] = 1.0
        return train_task(rid, head, est_s, **cfg)

    g0 = [run("r2-dense-s1", "dense", 1), run("r2-kasp-s1", "kasp", 1), run("r2-kasu-s1", "kasu", 1),
          run("r2-kasg-v1-s1", "kasg", 1, "v1"), run("r2-kasg-v3-s1", "kasg", 1, "v3"), run("r2-kas0-s1", "kas0", 1)]
    g1 = [run("r2-dense-s2", "dense", 2), run("r2-kasp-s2", "kasp", 2), run("r2-kasu-s2", "kasu", 2),
          run("r2-kasg-v2-s1", "kasg", 1, "v2"), run("r2-kasgshuf-v1-s1", "kasg_shuf", 1, "v1")]
    for g in (g0, g1):
        ids = [t["name"] for t in g]
        g.append({"type": "python", "name": "analysis-" + ids[0], "module": "kq5.analysis", "est_s": 600,
                  "kwargs": {"runs": ids, "split": "dev", "retok_rules": RETOK_RULES}})
    return {"name": "r2_pilot", "slug": "kq5-r2-pilot", "title": "kq5 r2 pilot",
            "deadline_s": 11.5 * 3600, "require_gpus": 2, "workers": {"0": g0, "1": g1},
            "backbone": backbone, "head_lr_mult": head_lr_mult}


K2_POINTS = [12_000_000, 24_000_000, 46_000_000, 68_000_000]


def r3(gen_variant: str, part: str, est: dict | None = None) -> dict:
    """Canonical runs: 100M tokens (1,526 steps x 65,536), dev evaluations at K2's 12/24/46/68M
    points (all dev windows, per-window sums saved), final dev + test, then dev and test
    analysis of each run right after it trains."""
    arch = BACKBONES["8L512"]
    g, glr = GEN_VARIANTS[gen_variant]
    common = dict(**arch, total_tokens=1526 * 65536, warmup_tokens=2 * 1048576, batch_seqs=64, micro_seqs=8,
                  eval_at_tokens=K2_POINTS, eval_windows_mid=None, final_dev_windows=None, final_test=True,
                  save_checkpoint=True, log_every=20, head_lr_mult=0.1)
    est = est or {"dense": 4300, "kasp": 4700, "kasu": 4700, "kas0": 4700, "kasg": 5500, "kasg_shuf": 5500}

    def pair(rid, head, seed):
        cfg = dict(common, seed=seed)
        if head == "dense":
            cfg["head_lr_mult"] = 1.0
        if head in ("kasg", "kasg_shuf"):
            cfg.update(generator=g, gen_lr_mult=glr)
        return [train_task(rid, head, est[head], **cfg),
                {"type": "python", "name": "analysis-" + rid, "module": "kq5.analysis", "est_s": 900,
                 "kwargs": {"runs": [rid], "splits": ["dev", "test"], "retok_rules": RETOK_RULES}}]

    def growth(runs):
        return {"type": "python", "name": "growth-" + runs[0], "module": "kq5.growth", "est_s": 1500,
                "kwargs": {"runs": runs}}

    if part == "x":     # arms that do not depend on the generator choice; launched before R2 is read
        g0 = pair("r3-dense-s1", "dense", 1) + pair("r3-kasp-s1", "kasp", 1) + pair("r3-kasu-s1", "kasu", 1)             + pair("r3-kas0-s1", "kas0", 1) + [growth(["r3-dense-s1", "r3-kasp-s1", "r3-kasu-s1", "r3-kas0-s1"])]
        g1 = pair("r3-dense-s2", "dense", 2) + pair("r3-kasp-s2", "kasp", 2) + pair("r3-kasu-s2", "kasu", 2)             + pair("r3-kas0-s2", "kas0", 2)
    elif part == "g":   # corrected arms, after the pilot (generator by Protocol 02 rule 3; u_zero init, 03b)
        def pz(rid, head, seed):
            tasks = pair(rid, head, seed)
            tasks[0]["config"]["corr_init"] = "u_zero"
            return tasks
        g0 = pz("r3-kasg-s1", "kasg", 1) + pz("r3-kasu0-s1", "kasu", 1) + pz("r3-kasgshuf-s1", "kasg_shuf", 1)             + [growth(["r3-kasg-s1", "r3-kasu0-s1", "r3-kasgshuf-s1"])]
        g1 = pz("r3-kasg-s2", "kasg", 2) + pz("r3-kasu0-s2", "kasu", 2) + pz("r3-kasgshuf-s2", "kasg_shuf", 2)             + [{"type": "python", "name": "headcost", "module": "kq5.headcost", "est_s": 1200,
                "kwargs": {"d_model": 512, "BT": 8192, "chunk": 32768}}]
    else:
        raise ValueError(part)
    return {"name": f"r3_canonical_{part}", "slug": f"kq5-r3-canonical-{part}", "title": f"kq5 r3 canonical {part}",
            "deadline_s": 11.6 * 3600, "require_gpus": 2, "workers": {"0": g0, "1": g1},
            "generator_variant": gen_variant}


def r4x() -> dict:
    """Re-analysis of kernel X's checkpoints with the final analysis code (after the independent
    review's fixes), so every arm is analysed by the same code as kernel G. No training."""
    def ana(rid):
        return {"type": "python", "name": "reanalysis-" + rid, "module": "kq5.analysis", "est_s": 900,
                "kwargs": {"runs": [rid], "splits": ["dev", "test"], "retok_rules": RETOK_RULES}}
    g0 = [ana(r) for r in ("r3-dense-s1", "r3-kasp-s1", "r3-kasu-s1", "r3-kas0-s1")] +         [{"type": "python", "name": "growth-x", "module": "kq5.growth", "est_s": 1500,
          "kwargs": {"runs": ["r3-dense-s1", "r3-kasp-s1", "r3-kas0-s1"]}}]
    g1 = [ana(r) for r in ("r3-dense-s2", "r3-kasp-s2", "r3-kasu-s2", "r3-kas0-s2")]
    return {"name": "r4x_reanalysis", "slug": "kq5-r4x-reanalysis", "title": "kq5 r4x reanalysis",
            "deadline_s": 3 * 3600, "require_gpus": 2, "workers": {"0": g0, "1": g1},
            "kernel_sources": ["KAGGLE_USER/kq5-r3-canonical-x"]}


def r1b() -> dict:
    """KAS-G throughput/memory after the chunked-generator fix (R1 OOM)."""
    w = {"0": [], "1": []}
    for gpu, bb in (("0", "8L512"), ("1", "6L768")):
        for gv in ("v1", "v3"):
            g, glr = GEN_VARIANTS[gv]
            w[gpu].append(train_task(f"r1b-tput-{bb}-kasg-{gv}", "kasg", 300, **BACKBONES[bb], generator=g,
                                     gen_lr_mult=glr, head_lr_mult=0.1, total_tokens=100_000_000,
                                     stop_tokens=40 * 65536, batch_seqs=64, micro_seqs=8, eval_at_tokens=[],
                                     final_dev_windows=8, save_checkpoint=False, log_every=5))
        w[gpu].append(train_task(f"r1b-tput-{bb}-kasp", "kasp", 300, **BACKBONES[bb], head_lr_mult=0.1,
                                 total_tokens=100_000_000, stop_tokens=40 * 65536, batch_seqs=64, micro_seqs=8,
                                 eval_at_tokens=[], final_dev_windows=8, save_checkpoint=False, log_every=5))
    return {"name": "r1b_kasg", "slug": "kq5-r1b-kasg", "title": "kq5 r1b kasg",
            "deadline_s": 3600, "require_gpus": 2, "workers": w}


def tpu_diag() -> dict:
    return {"name": "tpu_diag", "slug": "kq5-tpu-diagnostic", "title": "kq5 tpu diagnostic",
            "enable_gpu": False, "enable_tpu": True, "machine_shape": "TpuV5E8", "runner": "tpu_diag_runner.py",
            "deadline_s": 1800, "require_gpus": 0, "workers": {}}


if __name__ == "__main__":
    JOBS.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1]
    if which == "r2":
        spec = r2(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
    elif which == "r3":
        spec = r3(sys.argv[2] if sys.argv[3] == "g" else "v1", sys.argv[3])
    else:
        spec = {"r1": r1, "r1b": r1b, "r4x": r4x, "tpu": tpu_diag}[which]()
    path = JOBS / f"{spec['name']}.json"
    path.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(path)
