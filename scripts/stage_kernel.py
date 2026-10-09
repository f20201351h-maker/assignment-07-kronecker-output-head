"""Stage a private Kaggle script kernel: embed src/kq5 and a job spec into
scripts/kaggle_runner.py and write kernel-metadata.json.

    python scripts/stage_kernel.py configs/jobs/r1_smoke.json
    kaggle kernels push -p staging/kernels/<slug>
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USER = os.environ.get("KAGGLE_USERNAME", "KAGGLE_USER")  # Kaggle account that owns the dataset and kernels
DATASET = f"{USER}/kq5-fineweb-edu-tokens"


def src_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((ROOT / "src" / "kq5").rglob("*.py")):
            info = zipfile.ZipInfo(str(f.relative_to(ROOT / "src")).replace("\\", "/"),
                                   date_time=(2026, 1, 1, 0, 0, 0))
            z.writestr(info, f.read_bytes())
        for f in sorted((ROOT / "scripts" / "jobs").rglob("*.py")) if (ROOT / "scripts" / "jobs").exists() else []:
            info = zipfile.ZipInfo("jobs/" + f.name, date_time=(2026, 1, 1, 0, 0, 0))
            z.writestr(info, f.read_bytes())
    return buf.getvalue()


def main() -> None:
    spec_path = Path(sys.argv[1])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    slug = spec["slug"]
    out = ROOT / "staging" / "kernels" / slug
    out.mkdir(parents=True, exist_ok=True)
    z = src_zip()
    runner = (ROOT / "scripts" / spec.get("runner", "kaggle_runner.py")).read_text(encoding="utf-8")
    runner = runner.replace("__SRC_ZIP_B64__", base64.b64encode(z).decode()) \
                   .replace("__SPEC_JSON__", json.dumps(spec, indent=1))
    (out / "kernel.py").write_text(runner, encoding="utf-8")
    meta = {
        "id": f"{USER}/{slug}", "title": spec.get("title", slug.replace("-", " ")),
        "code_file": "kernel.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": spec.get("enable_gpu", True),
        "enable_tpu": spec.get("enable_tpu", False), "enable_internet": False,
        "machine_shape": spec.get("machine_shape", "NvidiaTeslaT4"),
        "dataset_sources": spec.get("dataset_sources", [] if spec.get("runner") else [DATASET]),
        "kernel_sources": spec.get("kernel_sources", []),
        "competition_sources": [], "model_sources": [],
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    record = {"slug": slug, "spec": str(spec_path), "src_zip_sha256": hashlib.sha256(z).hexdigest(),
              "kernel_sha256": hashlib.sha256((out / "kernel.py").read_bytes()).hexdigest()}
    (out / "stage_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    # tracked copy: the exact script Kaggle ran is part of the deliverable
    keep = ROOT / "kaggle" / slug
    keep.mkdir(parents=True, exist_ok=True)
    for name in ("kernel.py", "kernel-metadata.json", "stage_record.json"):
        (keep / name).write_bytes((out / name).read_bytes())
    (keep / "job_spec.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
