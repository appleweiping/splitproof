"""Download a few public Hugging Face datasets as Parquet and cache them locally.

Files go to ``$SPLITPROOF_DATA`` (default: ``~/.cache/splitproof``). Only the
files listed here are fetched; nothing is uploaded.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Any

BASE = "https://huggingface.co/datasets"
FILES = {
    "sms_spam": "ucirvine/sms_spam/resolve/main/plain_text/train-00000-of-00001.parquet",
    "emotion_train": "dair-ai/emotion/resolve/main/split/train-00000-of-00001.parquet",
    "emotion_test": "dair-ai/emotion/resolve/main/split/test-00000-of-00001.parquet",
    "go_emotions_train": (
        "google-research-datasets/go_emotions/resolve/main/simplified/train-00000-of-00001.parquet"
    ),
}


def path(name: str) -> Path:
    """Return the local Parquet file for ``name``, downloading it the first time."""
    cache = Path(os.environ.get("SPLITPROOF_DATA", Path.home() / ".cache" / "splitproof"))
    target = cache / f"{name}.parquet"
    if not target.exists():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}/{FILES[name]}"
        print(f"downloading {url}")
        partial = target.with_suffix(".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    return target


def columns(name: str) -> dict[str, list[Any]]:
    """Load a cached dataset as a dict of columns."""
    import pyarrow.parquet as pq

    return pq.read_table(path(name)).to_pydict()
