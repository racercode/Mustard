"""Atomic artifact writes and content digests."""
import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as file:
            name = file.name
            json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write('\n')
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()
