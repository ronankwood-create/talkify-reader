"""Verified local files for the optional ONNX Kokoro backend.

The ONNX backend needs two large files that are not in this repository - a
model graph and a voice pack. This module is the only thing allowed to decide
that a copy on disk is the file the manifest names.

The rule, and the reason for it
-------------------------------
**A file is used only when its SHA-256 matches a hash recorded in
``kokoro_onnx_assets.json``.** Not its size, not its name, not the fact that
the download reported success. A truncated or substituted model does not
crash - ONNX Runtime loads it and produces audio - so nothing downstream would
ever tell the user that what they are listening to came from the wrong graph.

A download never overwrites a known-good file. It lands in a temporary file
beside the target, is verified there, and only then is renamed into place. An
interrupted download therefore leaves the previous working asset untouched,
which matters because the alternative is a reader who closed a laptop lid mid
update and now has no working voice.

Recording a hash
----------------
The manifest ships with ``"sha256": null`` for both assets. That is not an
oversight and it is not a placeholder to be filled in with a guess: a hash
copied from anywhere other than a file that has actually been verified is
worse than no hash, because it looks like verification. Until a real hash is
recorded, this module refuses the asset and the ONNX backend stays off - the
PyTorch backend is unaffected, which is the whole point of them being separate
rollback switches.

To record one, run::

    python kokoro_assets.py record path\\to\\kokoro-v1.0.onnx

and paste the printed value into the manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(HERE, "kokoro_onnx_assets.json")

# Where verified assets live. Overridable so a machine with the models on
# another drive does not have to copy 300 MB into the source tree.
ASSET_DIR = os.environ.get("TALKIFY_KOKORO_ASSET_DIR") or os.path.join(
    HERE, "kokoro_assets")

CHUNK = 1 << 20


class AssetError(RuntimeError):
    """This asset cannot be trusted, so it is not used. Never fatal to
    Talkify: the caller falls back to the PyTorch backend."""


def load_manifest(path=None):
    """The checked-in manifest, or {} if it is missing or unreadable."""
    try:
        with open(path or MANIFEST_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    assets = data.get("assets") if isinstance(data, dict) else None
    return assets if isinstance(assets, dict) else {}


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path, entry):
    """Check one file against its manifest entry. Raises AssetError, or
    returns the verified hash.

    Size is checked first because it is free and rules out the common case
    (a truncated download) before hashing 300 MB. It is never sufficient on
    its own."""
    expected = entry.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise AssetError("no sha256 recorded for this asset in the manifest")
    try:
        size = os.path.getsize(path)
    except OSError as e:
        raise AssetError(f"asset is missing: {e}") from e
    wanted = entry.get("bytes")
    if isinstance(wanted, int) and wanted > 0 and size != wanted:
        raise AssetError(f"asset is {size} bytes, manifest says {wanted}")
    actual = sha256_of(path)
    if actual.lower() != expected.lower():
        raise AssetError("asset hash does not match the manifest")
    return actual


def verified_path(name, manifest=None, asset_dir=None):
    """The local path for one manifest asset, once it has been verified."""
    manifest = load_manifest() if manifest is None else manifest
    entry = manifest.get(name)
    if not isinstance(entry, dict):
        raise AssetError(f"no manifest entry for {name!r}")
    override = os.environ.get(entry.get("env") or "")
    path = override or os.path.join(asset_dir or ASSET_DIR,
                                    entry.get("file") or name)
    verify_file(path, entry)
    return path


def verified_paths(manifest=None, asset_dir=None):
    """(model, voices), both verified. Raises AssetError if either is not."""
    manifest = load_manifest() if manifest is None else manifest
    return (verified_path("model", manifest, asset_dir),
            verified_path("voices", manifest, asset_dir))


def identity(manifest=None):
    """A short stable name for exactly this pair of assets.

    Goes into the parent's cache identity: audio produced from one graph must
    never be served from a cache entry filled by another."""
    manifest = load_manifest() if manifest is None else manifest
    parts = []
    for name in ("model", "voices"):
        entry = manifest.get(name) or {}
        digest = entry.get("sha256")
        parts.append(digest[:12] if isinstance(digest, str) else "unset")
    return "-".join(parts)


def install(name, source_path, manifest=None, asset_dir=None):
    """Adopt a downloaded file as a verified asset.

    Verifies in place, in a temporary file beside the target, and only renames
    once the hash matches. A known-good asset is never removed or truncated
    first: if this fails, whatever was already working still works.
    """
    manifest = load_manifest() if manifest is None else manifest
    entry = manifest.get(name)
    if not isinstance(entry, dict):
        raise AssetError(f"no manifest entry for {name!r}")
    directory = asset_dir or ASSET_DIR
    target = os.path.join(directory, entry.get("file") or name)
    os.makedirs(directory, exist_ok=True)
    staging = target + ".incoming"
    if os.path.abspath(source_path) != os.path.abspath(staging):
        with open(source_path, "rb") as src, open(staging, "wb") as dst:
            while True:
                chunk = src.read(CHUNK)
                if not chunk:
                    break
                dst.write(chunk)
    try:
        verify_file(staging, entry)
    except AssetError:
        try:
            os.remove(staging)
        except OSError:
            pass
        raise
    os.replace(staging, target)
    return target


def _main(argv):
    if len(argv) == 3 and argv[1] == "record":
        print(sha256_of(argv[2]))
        return 0
    if len(argv) == 2 and argv[1] == "check":
        try:
            model, voices = verified_paths()
        except AssetError as e:
            print(f"onnx assets unavailable: {e}")
            return 1
        print(f"ok\n  {model}\n  {voices}")
        return 0
    print(__doc__.strip())
    print("\nusage: python kokoro_assets.py record <file> | check")
    return 2


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
