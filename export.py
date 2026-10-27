"""
Stitches a document's already-synthesized blocks into one contiguous MP3,
with word timings re-offset onto that single timeline. No ffmpeg anywhere,
so this only ever uses stdlib `wave` (to read PCM out of
Kokoro/stub's WAV blocks) and `lameenc` (a wheel-only MP3 encoder, no system
binary) to encode it.

    export_stream(blocks, mp3_path, json_path, mime)
        -> {"words", "duration", "audio_sha256", "timings_sha256",
            "block_durations"}
    assemble(blocks, mime) -> (mp3_bytes, words)

A single-file export writes the flat `[{w, s, e}]` array it always has. Pass
`chapter_index` and the sidecar becomes the commute package's object form
instead (see commute.py), which keeps each
word's source block - the thing a phone needs to resume on the right word and
the thing a flat array cannot say.

`blocks` is an *iterable* of (audio_bytes, words) per document block, in
reading order - `words` already a list of {"w","s","e"} dicts, not
engines.Word, so this module has no dependency on engines.py. `mime` is the
engine's mime for every block in the document (talkify.py never mixes engines
within one doc).

`export_stream` is the one to reach for: it consumes a generator and writes
straight to disk, so exporting a 900-page textbook holds one block's audio at
a time instead of the whole book's uncompressed WAV. `assemble` builds the
same bytes in memory and is kept for callers that want them as a value.

Both write through a `.part` file and `os.replace` onto the destination, so a
crash, a full disk or a killed process never leaves a half-written MP3 that
looks like a finished export.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import uuid
import wave
from pathlib import Path

import commute


def _offset_words(words, offset):
    return [{"w": w["w"], "s": w["s"] + offset, "e": w["e"] + offset} for w in words]


class _Hashed:
    """A write-through tee that hashes on the way past.

    The manifest has to state each asset's SHA-256, and a book-length MP3 is
    the one file big enough that reading it back to hash it would undo the
    streaming this module exists for. Hashing costs a gigabyte a second next
    to eight seconds of synthesis per block, so it is taken on the bytes as
    they are written and never mentioned again."""

    def __init__(self, handle, encode=None):
        self.handle = handle
        self.digest = hashlib.sha256()
        self._encode = encode

    def write(self, data):
        self.digest.update(self._encode(data) if self._encode else data)
        return self.handle.write(data)

    def hexdigest(self):
        return self.digest.hexdigest()

    # `_finish` and the failure path below drive the underlying handle
    # through these three, so the tee has to stand in for a file completely.
    def flush(self):
        self.handle.flush()

    def fileno(self):
        return self.handle.fileno()

    def close(self):
        self.handle.close()

    @property
    def closed(self):
        return self.handle.closed


class _ArrayTimings:
    """The legacy sidecar: one flat array of {w, s, e} for the whole file.

    Still what a single-file export writes, and what every reader built
    against Talkify so far expects."""

    def __init__(self, out):
        self.out = out
        self.count = 0

    def start(self):
        self.out.write("[")

    def block(self, index, words):
        for w in words:
            self.out.write(("," if self.count else "") + json.dumps(w))
            self.count += 1

    def close(self, duration):
        self.out.write("]")


class _ChapterTimings:
    """The chapter-package sidecar: an object that keeps block identity.

    A flat array cannot say which source block a word came from, and a phone
    resuming at "block 37, word 12" needs exactly that. Blocks stay in
    original order and each appears once, so this is the same words in the
    same order as the array form - grouped, and labelled with where they
    came from.

    `duration_ms` is written last because it is not known until the audio
    stream closes. JSON does not care about key order and this way nothing
    has to be buffered."""

    def __init__(self, out, chapter_index):
        self.out = out
        self.chapter_index = chapter_index
        self.count = 0
        self.blocks = 0

    def start(self):
        self.out.write('{"schema":"%s","chapter_index":%d,"blocks":['
                       % (commute.SCHEMA_CHAPTER_TIMINGS, self.chapter_index))

    def block(self, index, words):
        self.out.write('%s{"block_index":%d,"word_start_index":0,"words":['
                       % ("," if self.blocks else "", index))
        for i, w in enumerate(words):
            self.out.write(("," if i else "") + json.dumps(w))
            self.count += 1
        self.out.write("]}")
        self.blocks += 1

    def close(self, duration):
        self.out.write('],"duration_ms":%d}' % round(duration * 1000))


class _WavStream:
    """Kokoro and the stub engine emit WAV. Decode each block's PCM into the
    one running MP3 stream - one encoder per file, never per block, since
    independently encoded fragments concatenated together give players a
    wrong duration and broken seeking.

    Reports each block's duration as its *real* decoded length (frames/rate),
    not the last word's end time, which can sit a little short of the actual
    clip and would otherwise let timing drift accumulate block over block."""

    def __init__(self, out):
        self.out = out
        self.encoder = None

    def write(self, audio_bytes, words):
        import lameenc

        with wave.open(io.BytesIO(audio_bytes)) as w:
            rate, sampwidth, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
            nframes = w.getnframes()
            pcm = w.readframes(nframes)

        if self.encoder is None:
            if sampwidth != 2:
                raise ValueError(f"expected 16-bit PCM, got {sampwidth * 8}-bit")
            self.encoder = lameenc.Encoder()
            self.encoder.set_bit_rate(96)
            self.encoder.set_in_sample_rate(rate)
            self.encoder.set_channels(channels)
            self.encoder.set_quality(4)   # 0 (best/slowest) - 9 (worst/fastest); 4 is a fair middle ground

        self.out.write(self.encoder.encode(pcm))
        return nframes / rate

    def close(self):
        if self.encoder is not None:
            self.out.write(self.encoder.flush())


class _MpegStream:
    """Edge already returns MP3 per block. Raw concatenation of complete MP3
    frame streams (no ID3/Xing header from edge-tts) plays fine in practice -
    re-encoding would need to decode first, which is exactly the ffmpeg-sized
    dependency this project avoids. Block duration is the block's last word
    end, which is edge's own reported timing (not whisper-estimated), so it's
    already about as accurate as this gets without decoding."""

    def __init__(self, out):
        self.out = out

    def write(self, audio_bytes, words):
        self.out.write(audio_bytes)
        return words[-1]["e"] if words else 0.0

    def close(self):
        pass


def _stream_for(mime, out):
    if mime == "audio/wav":
        return _WavStream(out)
    if mime == "audio/mpeg":
        return _MpegStream(out)
    raise ValueError(f"don't know how to export audio of type {mime!r}")


def _part_path(destination):
    """A sibling of the destination, hidden by its leading dot, unique per
    attempt so two exports of the same document can never collide."""
    destination = Path(destination)
    return destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.part")


def _finish(handle, part, destination):
    handle.flush()
    os.fsync(handle.fileno())
    handle.close()
    os.replace(part, destination)


def export_stream(blocks, mp3_path, json_path, mime, *, id3=None,
                  chapter_index=None, block_start_index=0):
    """Encode `blocks` straight to disk as one MP3 plus its timings JSON.

    Nothing here accumulates the document: each block's audio is encoded and
    written as it arrives from the generator, and each block's words are
    appended to the JSON as they arrive too. Timings start at 0.0 for this
    file - so a chapter exported on its own gets its own timeline, exactly
    like the phone-side package expects.

    `chapter_index` selects the sidecar shape. Left out, the timings are the
    legacy flat array a single-file export has always written. Given, they are
    the commute package's object form, which records for every word which
    source block it came from - `block_start_index` is the document index of
    the first block in `blocks`, and the rest follow it in order.

    `id3` is a tag block written at the head of the MP3 before any audio. A
    chapter file gets one so an ID3-aware player shows a title instead of a
    filename; the single-file export passes none and its bytes are unchanged.

    Both files land through a `.part` file that is fsynced and then
    atomically renamed. Any failure - a generator that raises mid-book, a
    block that isn't 16-bit PCM, a disk that fills - deletes the parts and
    re-raises, so the export folder only ever holds whole files.

    Returns the word count, the real decoded duration in seconds, the SHA-256
    of each finished file, and each block's own duration - the last of which
    is what lets a caller say where block 37 starts on this timeline without
    decoding the MP3 again."""
    mp3_path, json_path = Path(mp3_path), Path(json_path)
    mp3_part, json_part = _part_path(mp3_path), _part_path(json_path)
    audio_out = words_out = None
    offset = 0.0
    durations = []
    try:
        audio_out = _Hashed(open(mp3_part, "xb"))
        words_out = _Hashed(open(json_part, "x", encoding="utf-8"),
                            encode=lambda s: s.encode("utf-8"))
        if id3:
            audio_out.write(id3)
        stream = _stream_for(mime, audio_out)
        timings = (_ArrayTimings(words_out) if chapter_index is None
                   else _ChapterTimings(words_out, chapter_index))
        timings.start()
        for n, (audio_bytes, words) in enumerate(blocks):
            duration = stream.write(audio_bytes, words)
            timings.block(block_start_index + n, _offset_words(words, offset))
            durations.append(duration)
            offset += duration
        stream.close()
        timings.close(offset)
        audio_sha, timings_sha = audio_out.hexdigest(), words_out.hexdigest()
        _finish(audio_out, mp3_part, mp3_path)
        _finish(words_out, json_part, json_path)
    except BaseException:
        for handle, part in ((audio_out, mp3_part), (words_out, json_part)):
            if handle is not None and not handle.closed:
                handle.close()
            part.unlink(missing_ok=True)
        raise
    return {"words": timings.count, "duration": offset,
            "audio_sha256": audio_sha, "timings_sha256": timings_sha,
            "block_durations": durations}


def assemble(blocks, mime):
    """The same encode, in memory. Kept for callers that want the bytes as a
    value rather than a file (and for anything already written against it)."""
    blocks = list(blocks)
    if not blocks:
        return b"", []
    out = io.BytesIO()
    stream = _stream_for(mime, out)
    combined_words = []
    offset = 0.0
    for audio_bytes, words in blocks:
        duration = stream.write(audio_bytes, words)
        combined_words.extend(_offset_words(words, offset))
        offset += duration
    stream.close()
    return out.getvalue(), combined_words
