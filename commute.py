"""
The commute package - what a phone actually receives.

This module is the format's
pure half. Everything here is a value in, a value out: hashes, slugs, the
manifest, the position sidecar and the ID3 tags. No engine, no session, no
HTTP - so smoke.py can assert the format directly instead of through an
export, and a future phone-side tool can import it without dragging Kokoro
along.

The one idea worth holding while reading it: **audio, timings and a position
always refer to one immutable rendition.** `rendition_id` is what makes that
checkable - change the text, the voice, the codec or the timing basis and it
changes too, so a sidecar written against yesterday's export cannot silently
move today's playback to the wrong place in the book. That is why the position
file carries the package's checksum rather than a checksum of itself.

Two rules that are easy to break by accident:

- **Canonical JSON is a wire format, not a pretty-printer.** Keys sorted
  recursively, compact separators, no trailing newline. Both hashes below are
  taken over its bytes, so reformatting the manifest by hand invalidates them.
- **The position file is the only mutable member of a package.** Everything
  else is written once and hashed. `write_json()` goes through a `.part`
  sibling and `os.replace`, so an interrupted write leaves the old complete
  file or the new one - never half of either.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import threading
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

_POSITION_LOCK = threading.Lock()

SCHEMA_PACKAGE = "talkify.commute-package/v1"
SCHEMA_POSITION = "talkify.position/v1"
SCHEMA_CHAPTER_TIMINGS = "talkify.chapter-timings/v1"
SCHEMA_INDEX = "talkify.export-index/v1"
TIMING_FORMAT = "talkify.word-timings/v1"
CHAPTER_LAYOUT = "talkify.chapters/v1"

PACKAGE_VERSION = 1
AUDIO_CODEC = "mp3"
AUDIO_BITRATE_KBPS = 96          # export.py's lameenc setting; part of the rendition

# Diagnostic only - the spec gives it no authority in conflict resolution, so
# it deliberately does not carry a username. This file syncs to Drive.
WRITER_ID = "talkify-desktop"

# Private to the desktop, and outside any package folder: a synchronizer may
# edit no package file but the position sidecar, so the table that maps a
# block index onto the export's millisecond timeline cannot live in one.
INDEX_NAME = ".talkify-packages.json"

DOC_SLUG_LIMIT = 80
CHAPTER_SLUG_LIMIT = 60


# ------------------------------------------------------------------ hashing

def canonical_json(obj):
    """UTF-8 JSON with keys sorted recursively, compact separators and no
    trailing newline. `ensure_ascii=False` matters: the spec says UTF-8, and
    a book title with an accent in it must hash the same here as it would in
    any other implementation of the spec."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def content_sha256(blocks):
    """Identifies the book's content, not its filename: source blocks in
    original order, two newlines between adjacent ones."""
    return sha256_hex("\n\n".join(blocks).encode("utf-8"))


def rendition_id(content_sha, voice):
    """Everything that would make already-encoded audio the wrong audio.

    A second export in another voice is a different rendition and gets its own
    position sidecar; it must never inherit the first one's place in the book.
    """
    return sha256_hex(canonical_json({
        "document_content_sha256": content_sha,
        "voice": voice or "",
        "audio_codec": AUDIO_CODEC,
        "audio_bitrate_kbps": AUDIO_BITRATE_KBPS,
        "timing_format": TIMING_FORMAT,
        "chapter_layout": CHAPTER_LAYOUT}))


def package_id_for(rendition):
    return f"talkify:{rendition}"


def package_checksum(rendition, content_sha, total_duration_ms, chapters):
    """Exactly the seven immutable members, per spec section 4 - display-only
    `document_title` is excluded, so renaming a book does not invalidate the
    sidecar sitting next to it."""
    return sha256_hex(canonical_json({
        "schema": SCHEMA_PACKAGE,
        "package_version": PACKAGE_VERSION,
        "package_id": package_id_for(rendition),
        "rendition_id": rendition,
        "document_content_sha256": content_sha,
        "total_duration_ms": total_duration_ms,
        "chapters": chapters}))


# -------------------------------------------------------------------- slugs

_DISALLOWED = re.compile(r"[^A-Za-z0-9 _-]+")
_WHITESPACE = re.compile(r"\s+")
_HYPHEN_RUN = re.compile(r"-{2,}")

# A file called NUL is not a file on Windows, it is the null device, and
# writing a package into one loses the whole export silently.
_DEVICE_NAMES = {"CON", "PRN", "AUX", "NUL",
                 *(f"COM{i}" for i in range(1, 10)),
                 *(f"LPT{i}" for i in range(1, 10))}


def slugify(title, limit=DOC_SLUG_LIMIT):
    """A filename stem that survives Drive, Windows and a media player's
    filename sort, per spec section 3.2.

    ASCII where possible means decompose-and-drop-the-accents, so "Café"
    becomes "Cafe" rather than "Caf" - dropping the base letter with its
    diacritic would mangle a title far worse than flattening it."""
    text = unicodedata.normalize("NFC", title or "")
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in decomposed if not unicodedata.combining(c))
    ascii_text = ascii_text.encode("ascii", "ignore").decode("ascii")
    slug = _DISALLOWED.sub("-", ascii_text)
    slug = _WHITESPACE.sub(" ", slug)
    slug = _HYPHEN_RUN.sub("-", slug)
    # Trimmed twice: truncating to the limit can leave a trailing separator
    # that the first trim had nothing to say about.
    slug = slug.strip(" .-")[:limit].strip(" .-")
    if not slug:
        return "untitled"
    if slug.upper() in _DEVICE_NAMES:
        return f"{slug}-doc"
    return slug


def chapter_slugs(titles):
    """One slug per chapter, all distinct. Two chapters really can share a
    title - "Problems", "Introduction" - and two files cannot share a name,
    so the second becomes `<slug>-2`."""
    used, out = {}, []
    for title in titles:
        base = slugify(title, CHAPTER_SLUG_LIMIT)
        key = base.lower()
        seen = used.get(key, 0) + 1
        used[key] = seen
        out.append(base if seen == 1 else f"{base}-{seen}")
    return out


def chapter_stem(ordinal, slug):
    """Three zero-padded digits first, because for a media player that ignores
    ID3 entirely the filename sort *is* the reading order."""
    return f"{ordinal:03d}_{slug}"


# ---------------------------------------------------------------- documents

def rfc3339_now():
    """UTC, millisecond precision, Z - the one format the spec accepts."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def build_manifest(title, rendition, content_sha, total_duration_ms, chapters):
    """The immutable `<document-slug>.package.json`.

    `chapters` is already in export order and already carries its own hashes;
    this only wraps it and takes the checksum over the result."""
    return {
        "schema": SCHEMA_PACKAGE,
        "package_version": PACKAGE_VERSION,
        "package_id": package_id_for(rendition),
        "rendition_id": rendition,
        "document_title": title,
        "document_content_sha256": content_sha,
        "total_duration_ms": total_duration_ms,
        "chapters": chapters,
        "package_checksum": {
            "algorithm": "sha256",
            "value": package_checksum(rendition, content_sha,
                                      total_duration_ms, chapters)}}


def manifest_chapter(chapter_index, ordinal, title, stem, global_start_ms,
                     duration_ms, block_start_index, block_end_index_exclusive,
                     audio_sha256, timings_sha256):
    """One chapter's row, with exactly the keys the checksum is taken over.
    Built here rather than inline so a key added to the manifest cannot go
    missing from the hash."""
    return {
        "chapter_index": chapter_index,
        "ordinal": ordinal,
        "title": title,
        "audio_file": f"{stem}.mp3",
        "timings_file": f"{stem}.json",
        "global_start_ms": global_start_ms,
        "duration_ms": duration_ms,
        "block_start_index": block_start_index,
        "block_end_index_exclusive": block_end_index_exclusive,
        "audio_sha256": audio_sha256,
        "timings_sha256": timings_sha256}


def position_document(rendition, checksum, elapsed_ms, block_index, word_index,
                      total_duration_ms, chapter_index=None,
                      chapter_elapsed_ms=None):
    """The one mutable file in a package.

    `checksum` is the manifest's `package_checksum.value` - a reference to
    immutable package contents, not a checksum of this file. A reader that
    finds it does not match what it is holding rejects the sidecar and keeps
    its own position rather than jumping somewhere wrong.

    `chapter_index` and `chapter_elapsed_ms` are written as a pair or not at
    all: the schema makes the second required whenever the first is present,
    and half of the pair is worse than neither."""
    doc = {
        "schema": SCHEMA_POSITION,
        "package_id": package_id_for(rendition),
        "rendition_id": rendition,
        "elapsed_ms": max(0, int(elapsed_ms)),
        "block_index": max(0, int(block_index)),
        "word_index": max(0, int(word_index)),
        "updated_at": rfc3339_now(),
        "total_duration_ms": max(0, int(total_duration_ms)),
        "checksum": {"algorithm": "sha256", "scope": SCHEMA_PACKAGE,
                     "value": checksum},
        "update_id": str(uuid.uuid4()),
        "writer_id": WRITER_ID}
    if chapter_index is not None and chapter_elapsed_ms is not None:
        doc["chapter_index"] = max(0, int(chapter_index))
        doc["chapter_elapsed_ms"] = max(0, int(chapter_elapsed_ms))
    return doc


def locate(chapters, elapsed_ms):
    """Which chapter a global millisecond lands in, and how far into it.

    Returns (chapter_index, chapter_elapsed_ms), or (None, None) with no
    chapters. A position exactly on a boundary belongs to the chapter that is
    ending, not the one about to start - spec section 7: "a location at
    chapter end remains in that chapter until the next track starts"."""
    if not chapters:
        return None, None
    elapsed_ms = max(0, int(elapsed_ms))
    for chapter in chapters:
        start = chapter["global_start_ms"]
        end = start + chapter["duration_ms"]
        if elapsed_ms <= end:
            return chapter["chapter_index"], max(0, elapsed_ms - start)
    last = chapters[-1]
    return last["chapter_index"], last["duration_ms"]


# ---------------------------------------------------------------- ID3 tags

def id3v2_3(tags):
    """A minimal ID3v2.3 tag block, to be written at the head of an MP3.

    Text frames only, UTF-16 with a BOM (encoding byte 1), which is the
    encoding every player has handled since 1999 and the only one in v2.3 that
    can hold a title with an em dash in it. The tag's own size is syncsafe -
    seven bits per byte - while each *frame* size in v2.3 is a plain big-endian
    integer. Getting those two the same way round is the classic way to
    produce a tag that one player reads and the next treats as garbage audio.
    """
    frames = b""
    for frame_id, value in tags.items():
        if not value:
            continue
        body = b"\x01\xff\xfe" + str(value).encode("utf-16-le") + b"\x00\x00"
        frames += (frame_id.encode("ascii") + struct.pack(">I", len(body))
                   + b"\x00\x00" + body)
    if not frames:
        return b""
    size = len(frames)
    syncsafe = bytes(((size >> 21) & 0x7F, (size >> 14) & 0x7F,
                      (size >> 7) & 0x7F, size & 0x7F))
    return b"ID3\x03\x00\x00" + syncsafe + frames


def chapter_id3(document_title, chapter_title, ordinal, chapter_count, stem,
                author=None):
    """The tag set from spec section 3.3. Filename order stays the fallback;
    this is what an ID3-aware player shows instead."""
    return id3v2_3({
        "TALB": document_title,
        "TPE2": author or "Talkify",
        "TIT2": chapter_title,
        "TRCK": f"{ordinal}/{chapter_count}",
        "TPOS": "1/1",
        "TSOT": stem})


# ------------------------------------------------------------------ writing

def write_json(path, obj, part_stem=None):
    """Serialize, fsync, then `os.replace` onto the destination.

    Interrupting this leaves the old complete JSON or the new complete JSON,
    which is acceptance check 5. The part file is a hidden sibling in the same
    directory, because `os.replace` is only atomic within one filesystem, and
    it carries a uuid so two writers cannot collide on it. Readers ignore
    `.part`."""
    path = Path(path)
    stem = part_stem or path.name
    part = path.with_name(f".{stem}.{uuid.uuid4().hex}.part")
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    handle = open(part, "xb")
    try:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(part, path)
    except BaseException:
        if not handle.closed:
            handle.close()
        part.unlink(missing_ok=True)
        raise
    return path


def position_path(directory, slug):
    return Path(directory) / f"{slug}.position.json"


def write_position(directory, slug, document):
    """Writes `<slug>.position.json` through `.<slug>.position.<uuid>.part`,
    the name spec section 6.5 asks for."""
    path = position_path(directory, slug)
    return write_json(path, document, part_stem=f"{slug}.position")


# ------------------------------------------------------- the desktop's index

def read_index(export_dir):
    """The desktop's private map from document content to the package it was
    exported into. Missing or malformed reads as empty: a corrupt index must
    cost the reader nothing, since the only thing it can do is skip a sidecar
    update."""
    path = Path(export_dir) / INDEX_NAME
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("schema") != SCHEMA_INDEX:
        return {}
    packages = data.get("packages")
    return packages if isinstance(packages, dict) else {}


def write_index(export_dir, packages):
    return write_json(Path(export_dir) / INDEX_NAME,
                      {"schema": SCHEMA_INDEX, "packages": packages},
                      part_stem=INDEX_NAME.lstrip("."))


def record_package(export_dir, content_sha, entry):
    """Remember one export, replacing whatever was recorded for the same
    content. A re-export in a second voice is a new rendition and supersedes
    the first: there is one current package per document, and the position
    heartbeat updates that one."""
    packages = read_index(export_dir)
    packages[content_sha] = entry
    write_index(export_dir, packages)
    return entry


def elapsed_for(entry, block_index, within_seconds):
    """Where (block, within) lands on the export's timeline, in milliseconds.

    `block_starts_ms` holds one entry per exported block plus a final sentinel
    equal to the package duration, so a block's own length is the gap to the
    next entry and the arithmetic needs no special case for the last block.
    `within` is clamped into the block it names: a heartbeat can arrive with an
    audio offset a shade past the end of a block, and letting that leak forward
    would put the sidecar in the next chapter."""
    starts = entry.get("block_starts_ms") or []
    offset = int(entry.get("block_offset", 0))
    i = int(block_index) - offset
    if not starts or not (0 <= i < len(starts) - 1):
        return None
    start, end = starts[i], starts[i + 1]
    return max(start, min(end, start + int(round(float(within_seconds) * 1000))))

