import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOKENIZER = ROOT / "data" / "gpt2" / "tokenizer.json"


@pytest.fixture(scope="session")
def gpt2_bytes():
    if not TOKENIZER.exists():
        pytest.skip("GPT-2 tokenizer.json not downloaded")
    from kq5.gpt2bytes import load_gpt2_raw_bytes
    return load_gpt2_raw_bytes(TOKENIZER)


@pytest.fixture(scope="session")
def tokenizer():
    if not TOKENIZER.exists():
        pytest.skip("GPT-2 tokenizer.json not downloaded")
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER))
