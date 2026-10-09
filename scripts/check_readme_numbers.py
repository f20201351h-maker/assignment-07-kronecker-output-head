"""Check that every number in a document is backed by a results file or an explicit allowlist.

A number passes if (a) some numeric leaf of a JSON file under results/ (or experiments/**/summary.json)
rounds to it at the printed precision (also as a percentage, as an absolute value, or in millions/thousands when written
with M/K), or (b) it is listed in
scripts/number_allowlist.json with a reason (configuration constants, years, arXiv ids,
values quoted from cited sources). Exit code 1 if anything is unmatched.

    python scripts/check_readme_numbers.py [README.md | docs/EXPERIMENTS.md | docs/EXPRESSIVITY.md]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\w.])[-−+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?\s?(?:M|K|k|%|×)?(?![\w])")


def leaves(obj, out):
    if isinstance(obj, dict):
        for v in obj.values():
            leaves(v, out)
    elif isinstance(obj, list):
        for v in obj:
            leaves(v, out)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        out.append(float(obj))


def result_values() -> list[float]:
    vals: list[float] = []
    files = list((ROOT / "results").rglob("*.json")) + list((ROOT / "experiments").rglob("summary.json"))
    for f in files:
        try:
            leaves(json.loads(f.read_text(encoding="utf-8")), vals)
        except Exception:  # noqa: BLE001
            pass
    return vals


def parse(token: str):
    t = token.replace("−", "-").replace(",", "").strip()
    suffix = ""
    if t and t[-1] in "MKk%×":
        suffix = t[-1]
        t = t[:-1].strip()
    decimals = len(t.split(".")[1]) if "." in t and "e" not in t.lower() else 0
    return float(t), decimals, suffix


def matches(value: float, decimals: int, suffix: str, pool: list[float]) -> bool:
    tol = 0.5 * 10 ** (-decimals) + 1e-12
    scales = [1.0]
    if suffix == "%":
        scales = [100.0, 1.0]
    elif suffix == "M":
        scales = [1e-6]
    elif suffix in ("K", "k"):
        scales = [1e-3]
    for v in pool:
        for s in scales:
            x = v * s
            if abs(x - value) <= tol or abs(abs(x) - abs(value)) <= tol:
                return True
    return False


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    readme = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = re.sub(r"\]\([^)]*\)", "]", text)              # drop link targets (paths, urls)
    text = re.sub(r"```.*?```", " ", text, flags=re.S)     # code blocks (before inline code)
    text = re.sub(r"`[^`\n]*`", " ", text)                 # inline code: identifiers, commands
    text = re.sub(r"(?m)^(#+\s*)[\d.–-]+", r"", text)    # section numbers in headings
    text = re.sub(r"(?m)^(\s*)\d+\.\s", r"", text)       # numbered-list markers
    allow_path = ROOT / "scripts" / "number_allowlist.json"
    allow = json.loads(allow_path.read_text(encoding="utf-8")) if allow_path.exists() else []
    allowed = {a["value"] for a in allow}
    pool = result_values()
    unmatched, checked = [], 0
    for m in NUM.finditer(text):
        tok = m.group(0).strip()
        try:
            value, dec, suf = parse(tok)
        except ValueError:
            continue
        checked += 1
        norm = tok.replace("−", "-").replace(" ", "")
        bare = norm.rstrip("MKk%×,")
        if norm in allowed or tok in allowed or bare in allowed:
            continue
        if matches(value, dec, suf, pool):
            continue
        line = text.count("\n", 0, m.start()) + 1
        unmatched.append({"token": tok, "line_in_stripped_text": line,
                          "context": text[max(0, m.start() - 60):m.end() + 40].replace("\n", " ")})
    print(f"checked {checked} numbers; unmatched {len(unmatched)}")
    for u in unmatched[:60]:
        print(f"  {u['token']!r}: ...{u['context']}...")
    return 1 if unmatched else 0


if __name__ == "__main__":
    sys.exit(main())
