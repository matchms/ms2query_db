from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return [jsonable(x) for x in value]
    if type(value).__name__ in {"NAType", "NaTType"}:
        return None
    return str(value)


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def scientific_versions():
    result = {}
    for name in ("matchms", "ms2deepscore", "chemap", "rdkit", "numpy"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as out:
            temp = Path(out.name)
            json.dump(jsonable(data), out, sort_keys=True, indent=2, allow_nan=False)
            out.write("\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def text_value(value):
    value = jsonable(value)
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"nan", "none", "null", "n/a", "<na>"} else value


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]
